"""Phase 1 + 2 evaluation of the trained models in models/ on train / validation / test (no training).

    python scripts/04_evaluate.py              # detector + both recognizers, all splits
    python scripts/04_evaluate.py --skip-train-split   # faster: validation + test only

Run 01_prepare_data.py first (it recreates the exact splits of the trained models).
Writes reports/evaluation/: metrics CSVs, comparison table, figures, run_log.txt.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from platereader import data_lpr as L, evaluate as EV  # noqa: E402
from platereader.config import add_common_args, config_from_args  # noqa: E402
from platereader.train_detector import det_metrics  # noqa: E402
from platereader.utils import FigureSaver, RunLog, free_gpu, pick_device, set_seed, yolo_device  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--skip-train-split", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = cfg.paths
    set_seed(cfg.seed)
    out = P.reports("evaluation")
    log, save = RunLog(out / "run_log.txt"), FigureSaver(out / "figures")
    device = pick_device(args.device)
    splits = ["val", "test"] if args.skip_train_split else ["train", "val", "test"]

    # ---------------- detector
    from ultralytics import YOLO

    import json
    imgsz = json.loads((P.models_dir / "lpr_config.json").read_text(encoding="utf-8"))["yolo_imgsz"]
    det = YOLO(str(P.models_dir / "lpd_best.pt"))
    rows = []
    for s in splits:
        m = det.val(data=str(P.work_dir / "lpd_data.yaml"), split=s, imgsz=imgsz, batch=8, device=yolo_device(device),
                    workers=0, project=str(P.work_dir / "eval_val"), name=s, exist_ok=True, plots=False, verbose=False)
        rows.append({"split": s, **det_metrics(m)})
        free_gpu()
    det_tab = pd.DataFrame(rows).set_index("split")
    det_tab.to_csv(out / "split_metrics_detector.csv")
    log("detector per split:\n" + det_tab.round(4).to_string())
    if "train" in det_tab.index:
        EV.plot_split_bars(det_tab, ["precision", "recall", "mAP50", "mAP50-95"], ["precision", "recall", "mAP50", "mAP50-95"],
                           "YOLO26 detector: train vs validation vs test", "eval_detector_train_val_test", save)
        log("detector mAP50:", EV.verdict(*det_tab["mAP50"].reindex(["train", "val", "test"])))

    # ---------------- recognizers
    vocab = (P.splits_dir / "letter_vocab.txt").read_text(encoding="utf-8").split("\n")
    enc = L.Encoder(vocab)
    img_dir = P.lpr_dir / "detections"
    dfs = {s: pd.read_csv(P.splits_dir / f"lpr_{s}_split.csv", dtype=str, encoding="utf-8").rename(columns={"label": "text"})
           for s in splits}
    loaders = {s: L.make_loader(L.PLPRDataset(d, img_dir, enc), 128, False) for s, d in dfs.items()}
    ckpts = {n: P.models_dir / f"lpr_{n}_best.pt" for n in ("scratch", "pretrained")}
    models, test_res, rec_rows = {}, {}, []
    for name, path in ckpts.items():
        model, _ = EV.load_recognizer(path, device)
        models[name] = model
        for s in splits:
            r, Pp, Y = EV.recognizer_metrics(model, loaders[s], device, enc)
            rec_rows.append({"model": name, "split": s, "samples": len(Y), **{k: v for k, v in r.items() if k != "pos_acc"}})
            if s == "test":
                test_res[name] = r
                pd.DataFrame({"filename": dfs[s].filename, "true": [enc.decode(t) for t in Y], "pred": [enc.decode(t) for t in Pp],
                              "correct": (Pp == Y).all(1).numpy()}).to_csv(out / f"lpr_{name}_test_predictions.csv", index=False, encoding="utf-8")
                EV.plot_letter_confusion(Pp, Y, vocab, name, save)
                EV.plot_misclassified(Pp, Y, dfs[s].filename.tolist(), img_dir, enc, name, save)
        free_gpu()
    rec_tab = pd.DataFrame(rec_rows).set_index(["model", "split"])
    rec_tab.to_csv(out / "split_metrics_recognizers.csv")
    log("recognizers per split:\n" + rec_tab.round(4).to_string())
    for name in models:
        t = rec_tab.loc[name]
        if "train" in t.index:
            EV.plot_split_bars(t, ["plate_acc", "char_acc", "letter_acc", "digit_macro_f1", "letter_macro_f1"],
                               ["full-plate acc", "per-char acc", "letter acc", "digit macro F1", "letter macro F1"],
                               f"{name} recognizer: train vs validation vs test", f"eval_{name}_train_val_test", save)
            log(f"{name} full-plate accuracy:", EV.verdict(*t["plate_acc"].reindex(["train", "val", "test"])))

    comp = EV.comparison_table(models, test_res, ckpts, device, cfg["lpr"]["timing"])
    comp.to_csv(out / "lpr_comparison.csv")
    log("comparison:\n" + comp.T.to_string())
    log(f"done: results in {out}")


if __name__ == "__main__":
    main()
