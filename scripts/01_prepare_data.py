"""Inspect both datasets, draw the EDA figures, create the train/val/test splits and the YOLO layout.

    python scripts/01_prepare_data.py            # reports/data/ + work/splits/ + work/lpd_yolo/
    python scripts/01_prepare_data.py --smoke    # same with tiny subsets (for a quick training check)

The split is deterministic (seed 42) and reproduces the split of the models in models/splits/; this is verified.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from platereader import data_lpd as D, data_lpr as L  # noqa: E402
from platereader.config import add_common_args, config_from_args  # noqa: E402
from platereader.utils import FigureSaver, RunLog  # noqa: E402


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args)
    P = cfg.paths
    out = P.reports("data")
    log, save = RunLog(out / "run_log.txt"), FigureSaver(out / "figures")

    # ---------------- LPD (detection)
    scan = D.scan_lpd(P.lpd_dir, log=log)
    D.plot_lpd_eda(scan, save, seed=cfg.seed)
    splits, n_groups = D.group_split(scan.valid, scan.plates, cfg["lpd"]["split"], cfg.seed)
    log(f"LPD split: {n_groups} vehicle groups ->", {k: len(v) for k, v in splits.items()})
    D.write_split_files(splits, scan.images, P.splits_dir)
    ref = D.read_split_files(P.models_dir / "splits")
    if ref:
        same = all(ref[k] == splits[k] for k in ref)
        log("split identical to the one of the trained models (models/splits):", same)
        if not same:
            log("WARNING: split differs from models/splits -> test metrics of models/ would include training images")
    n = cfg["lpd"]["smoke_n"]
    use = {k: v[: n[k]] for k, v in splits.items()} if n else splits
    D.build_yolo_layout(use, scan.images, scan.clean_labels, P.lpd_yolo_dir, P.lpd_dir, log=log)
    D.write_data_yaml(P.lpd_yolo_dir, P.work_dir / "lpd_data.yaml")

    # ---------------- LPR (recognition)
    df, letter_vocab = L.load_lpr(P.lpr_dir, log=log)
    L.plot_lpr_eda(df, letter_vocab, P.lpr_dir, save, seed=cfg.seed)
    lpd_test_plates = {t for s in splits["test"] for t in scan.plates.get(s, [])}
    df = L.split_lpr(df, lpd_test_plates, cfg["lpr"]["split"]["holdout"], cfg.seed, log=log)
    for s in ("train", "val", "test"):
        df.loc[df.split == s, ["filename", "text"]].rename(columns={"text": "label"}).to_csv(
            P.splits_dir / f"lpr_{s}_split.csv", index=False, encoding="utf-8")
    ref_lpr = P.models_dir / "splits" / "lpr_test_split.csv"
    if ref_lpr.exists():
        import pandas as pd

        same = set(pd.read_csv(ref_lpr, dtype=str).filename) == set(df.filename[df.split == "test"])
        log("LPR test split identical to models/splits:", same)
    (P.splits_dir / "letter_vocab.txt").write_text("\n".join(letter_vocab), encoding="utf-8")
    log(f"done: splits in {P.splits_dir}, YOLO layout in {P.lpd_yolo_dir}, figures in {out / 'figures'}")


if __name__ == "__main__":
    main()
