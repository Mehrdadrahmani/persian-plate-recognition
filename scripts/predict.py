"""Read licence plates from images on the command line.

    python scripts/predict.py car.jpg other.jpg                 # print the plates
    python scripts/predict.py photos/*.jpg --json               # one JSON line per image
    python scripts/predict.py car.jpg --save-vis out/           # save images with boxes + Persian plate text
    python scripts/predict.py car.jpg --backend onnx            # ONNX Runtime instead of PyTorch
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from platereader.config import REPO_ROOT  # noqa: E402
from platereader.pipeline import PlateRecognizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None, help="detector threshold (default: from lpr_config.json)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--save-vis", default=None, help="folder for annotated images")
    args = ap.parse_args()
    reader = PlateRecognizer(args.models_dir, backend=args.backend, device=args.device, conf=args.conf)
    for path in args.images:
        plates = reader.predict(path)
        if args.save_vis:
            Path(args.save_vis).mkdir(parents=True, exist_ok=True)
            Image.fromarray(reader.draw(path, plates)).save(Path(args.save_vis) / Path(path).name)
        if args.json:
            print(json.dumps({"image": path, "plates": [p.to_dict() for p in plates]}, ensure_ascii=False))
        else:
            print(f"{path}: " + (", ".join(f"{p.persian}  ({p.latin}, reading confidence {p.text_conf:.2f})" for p in plates)
                                  or "no plate found"))


if __name__ == "__main__":
    main()
