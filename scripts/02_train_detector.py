"""Phase 1: train the YOLO26 plate detector and evaluate it on the validation and test splits.

    python scripts/02_train_detector.py            # full training (use a CUDA GPU; ~1 h on 2 x T4)
    python scripts/02_train_detector.py --smoke    # 1 epoch on a few images: checks that everything runs

Run 01_prepare_data.py first. Resumes automatically from work/runs/lpd/<name>/weights/last.pt.
The trained weights stay in work/runs; use 05_export_bundle.py to update models/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from platereader import train_detector as TD  # noqa: E402
from platereader.config import add_common_args, config_from_args  # noqa: E402
from platereader.utils import FigureSaver, RunLog, pick_device, set_seed, yolo_device  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    cfg = config_from_args(args)
    P, C = cfg.paths, cfg["lpd"]
    set_seed(cfg.seed)
    out = P.reports("detector")
    log, save = RunLog(out / "run_log.txt"), FigureSaver(out / "figures")
    device = pick_device(args.device)
    data_yaml = P.work_dir / "lpd_data.yaml"
    assert data_yaml.exists(), "run scripts/01_prepare_data.py first"

    from ultralytics import YOLO

    weights_dir = P.work_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    name = f"{Path(C['model']).stem}_{P.tag}"
    run_dir, best = TD.train_detector(weights_dir / C["model"], data_yaml, P.runs_dir / "lpd", name, C["train"],
                                      yolo_device(device, all_gpus=True), workers=args.workers if device.type == "cuda" else 0,
                                      seed=cfg.seed, log=log)
    log("best weights:", best)
    TD.plot_training_curves(run_dir / "results.csv", save)

    model = YOLO(str(best))
    kw = dict(data=str(data_yaml), imgsz=C["train"]["imgsz"], batch=C["train"]["batch"], device=yolo_device(device),
              workers=0, project=str(P.runs_dir / "lpd"), exist_ok=True, plots=True, verbose=False)
    m_val = model.val(split="val", name=f"{name}_val", **kw)
    m_test = model.val(split="test", name=f"{name}_test", **kw)
    conf = TD.best_f1_conf(m_val)
    metrics = {"val": TD.det_metrics(m_val), "test": TD.det_metrics(m_test), "recommended_conf_threshold": conf,
               "imgsz": C["train"]["imgsz"], "run": name}
    (out / "lpd_metrics.json").write_text(json.dumps(metrics, indent=2))
    log("detector metrics:", json.dumps(metrics))
    TD.show_ultralytics_plots({"val": m_val.save_dir, "test": m_test.save_dir}, save)
    test_imgs = sorted((P.lpd_yolo_dir / "images" / "test").iterdir())
    TD.detector_error_analysis(model, test_imgs, P.lpd_yolo_dir / "labels" / "test", C["train"]["imgsz"],
                               yolo_device(device), conf, out, save, log=log)
    log(f"done: results in {out}")


if __name__ == "__main__":
    main()
