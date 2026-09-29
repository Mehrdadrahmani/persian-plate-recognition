"""Phase 3: evaluate the complete pipeline (YOLO26 detection -> crop -> CNN recognition) on the held-out test split.

    python scripts/06_evaluate_end_to_end.py                 # all 938 test images, PyTorch backend
    python scripts/06_evaluate_end_to_end.py --backend onnx  # same with ONNX Runtime (what the Android app runs)
    python scripts/06_evaluate_end_to_end.py --limit 50      # quick check

Writes reports/end_to_end/: end_to_end_metrics.json, CSV tables, run_log.txt and figures/*.png.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from platereader import e2e_eval as E  # noqa: E402
from platereader.config import add_common_args, config_from_args  # noqa: E402
from platereader.data_lpd import write_data_yaml  # noqa: E402
from platereader.pipeline import PlateRecognizer  # noqa: E402
from platereader.train_detector import det_metrics, show_ultralytics_plots  # noqa: E402
from platereader.utils import FigureSaver, RunLog, pick_device, set_seed, yolo_device  # noqa: E402


def detection_map_on_test(cfg, test_files, device, out_dir, log):
    """Standard detection metrics (mAP50, mAP50-95, P, R) of the detector on the test split via Ultralytics."""
    from ultralytics import YOLO

    lay = cfg.paths.work_dir / "e2e_test_yolo"
    if lay.exists():
        shutil.rmtree(lay)
    (lay / "images" / "test").mkdir(parents=True)
    (lay / "labels" / "test").mkdir(parents=True)
    for f in test_files:
        (lay / "images" / "test" / f).symlink_to((cfg.paths.lpd_dir / "images" / f).resolve())
        shutil.copy2(cfg.paths.lpd_dir / "labels" / f"{Path(f).stem}.txt", lay / "labels" / "test")
    yaml_path = write_data_yaml(lay, lay / "data.yaml")
    yaml_path.write_text(yaml_path.read_text().replace("images/train", "images/test").replace("images/val", "images/test"))
    imgsz = json.loads((cfg.paths.models_dir / "lpr_config.json").read_text(encoding="utf-8"))["yolo_imgsz"]
    m = YOLO(str(cfg.paths.models_dir / "lpd_best.pt")).val(
        data=str(yaml_path), split="test", imgsz=imgsz, batch=8, device=yolo_device(device), workers=0,
        project=str(cfg.paths.work_dir / "e2e_val"), name="test", exist_ok=True, plots=True, verbose=False)
    res = det_metrics(m)
    log("detector on the test split (Ultralytics val):", {k: round(v, 4) for k, v in res.items()})
    return res, m.save_dir


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N test images")
    ap.add_argument("--skip-map", action="store_true", help="skip the Ultralytics mAP evaluation")
    args = ap.parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    out = cfg.paths.reports("end_to_end")
    log, save = RunLog(out / "run_log.txt"), FigureSaver(out / "figures")
    device = pick_device(args.device)
    test_files = (cfg.paths.models_dir / "splits" / "lpd_test_split.txt").read_text().split()[: args.limit]
    log(f"===== Phase 3 end-to-end evaluation | backend={args.backend} device={device} | {len(test_files)} test images")

    det_map, val_dir = ({}, None) if args.skip_map else detection_map_on_test(cfg, test_files, device, out, log)
    if val_dir:
        show_ultralytics_plots({"test": val_dir}, save)

    reader = PlateRecognizer(cfg.paths.models_dir, backend=args.backend, device=args.device)
    items = E.load_ground_truth(cfg.paths.lpd_dir, test_files)
    cache, timing = E.run_pipeline(reader, items, log=log)

    sweep = E.threshold_sweep(items, cache)
    sweep.to_csv(out / "e2e_threshold_sweep.csv", index=False)
    best_thr = float(sweep.loc[sweep.e2e_f1.idxmax(), "threshold"])
    trained_thr = reader.conf
    E.plot_sweep(sweep, best_thr, trained_thr, save)

    results = {}
    for name, thr in (("phase1_threshold", trained_thr), ("best_f1_threshold", best_thr)):
        m, plates, imgs = E.evaluate_at(items, cache, thr)
        results[name] = m
        log(f"--- {name} = {thr:.3f}: " + ", ".join(f"{k} {m[k]:.4f}" for k in (
            "det_precision", "det_recall", "e2e_precision", "e2e_recall", "e2e_f1", "recognition_acc_on_detected",
            "cer_on_detected", "image_acc")))
        if name == "best_f1_threshold":
            plates.to_csv(out / "e2e_plates.csv", index=False, encoding="utf-8")
            imgs.to_csv(out / "e2e_images.csv", index=False)
            E.plot_summary_bars(m, save)
            parts = E.plot_error_breakdown(m, plates, save)
            log("error breakdown:", parts)
            E.plot_position_accuracy(m, save)
            E.plot_accuracy_by_size(plates, save).to_csv(out / "e2e_accuracy_by_plate_height.csv", index=False)
            E.plot_letter_confusion(plates, reader.letters, save)
            E.plot_gallery(items, plates, save, correct=True)
            E.plot_gallery(items, plates, save, correct=False)
            log("missed plates at this threshold:", E.plot_missed(items, cache, thr, save))

    import pandas as pd

    pd.DataFrame(results).T.rename_axis("operating_point").to_csv(out / "e2e_summary.csv")
    summary = {"backend": args.backend, "device": str(device), "test_images": len(items), "detector_test_map": det_map,
               "speed_ms_per_image": timing, "recommended_threshold": best_thr, "phase1_threshold": trained_thr,
               "end_to_end": results}
    (out / "end_to_end_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    b = results["best_f1_threshold"]
    log(f"===== DONE: end-to-end F1 {b['e2e_f1']:.4f} (P {b['e2e_precision']:.4f}, R {b['e2e_recall']:.4f}) at threshold "
        f"{best_thr:.2f}; {b['correct_reads']}/{b['gt_plates']} test plates read exactly; results in {out}")


if __name__ == "__main__":
    main()
