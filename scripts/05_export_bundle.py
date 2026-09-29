"""Export freshly trained models (work/runs) into an inference bundle (models/ by default).

    python scripts/05_export_bundle.py --best scratch --force        # overwrite models/ with the new training run
    python scripts/05_export_bundle.py --smoke                       # -> work/bundle_smoke (never touches models/)

The bundle holds: lpd_best.pt/.onnx, lpr_best.ts/.onnx, lpr_<arch>_best.pt, lpr_config.json, splits/.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from platereader import export as X  # noqa: E402
from platereader.config import add_common_args, config_from_args  # noqa: E402
from platereader.utils import RunLog  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--best", default="scratch", choices=["scratch", "pretrained"], help="recognizer to ship")
    ap.add_argument("--out", default=None, help="output folder (default: models/, or work/bundle_smoke with --smoke)")
    ap.add_argument("--force", action="store_true", help="allow overwriting models/")
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = cfg.paths
    out = Path(args.out) if args.out else (P.work_dir / "bundle_smoke" if cfg.smoke else P.models_dir)
    if out.resolve() == P.models_dir.resolve() and any(out.glob("*.pt")) and not args.force:
        sys.exit(f"{out} already has models; pass --force to overwrite (or --out another folder)")
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(out / "export_log.txt")

    name = f"{Path(cfg['lpd']['model']).stem}_{P.tag}"
    det_best = P.runs_dir / "lpd" / name / "weights" / "best.pt"
    imgsz = cfg["lpd"]["train"]["imgsz"]
    X.export_detector(det_best, out, imgsz, log)
    for arch in ("scratch", "pretrained"):
        src = P.runs_dir / "lpr" / f"lpr_{arch}_best.pt"
        if src.exists():
            shutil.copy2(src, out / src.name)
    ck = X.export_recognizer(P.runs_dir / "lpr" / f"lpr_{args.best}_best.pt", out, log=log)
    metrics = P.reports("detector") / "lpd_metrics.json"
    conf = json.loads(metrics.read_text())["recommended_conf_threshold"] if metrics.exists() else 0.25
    X.write_config(out, ck, args.best, imgsz, conf, cfg["pipeline"]["crop_padding"])
    (out / "splits").mkdir(exist_ok=True)
    for f in P.splits_dir.glob("lpd_*_split.txt"):
        shutil.copy2(f, out / "splits" / f.name)
    for f in P.splits_dir.glob("lpr_*_split.csv"):
        shutil.copy2(f, out / "splits" / f.name)
    log(f"bundle ready in {out}")


if __name__ == "__main__":
    main()
