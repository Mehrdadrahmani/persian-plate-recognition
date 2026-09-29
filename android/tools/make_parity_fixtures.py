"""Reference outputs of the Python ONNX pipeline (with the shipped mobile models) for the Kotlin parity tests (android/core/src/test).

    python android/tools/make_parity_fixtures.py        # -> android/core/src/test/resources/parity.json

For N LPD test images: plates found by the full pipeline (text, box, confidences).
For M LPR test crops: recognizer output text + confidence.
For a few crops: the exact letterboxed 64x256 uint8 pixels (checks the Pillow-compatible resize in Kotlin).
"""
import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from platereader import recognizer as R  # noqa: E402
from platereader.pipeline import PlateRecognizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=40)
    ap.add_argument("--crops", type=int, default=200)
    args = ap.parse_args()
    mob = json.loads((ROOT / "models" / "mobile" / "mobile_config.json").read_text())  # what the app ships
    det_rel, rec_rel = "mobile/lpd_mobile.onnx", "mobile/lpr_mobile.onnx"
    reader = PlateRecognizer(ROOT / "models", backend="onnx", det_model=ROOT / "models" / det_rel,
                             rec_model=ROOT / "models" / rec_rel, imgsz=mob["imgsz"], conf=mob["conf"])
    lpd = ROOT / "data" / "LPD" / "images"
    lpr = ROOT / "data" / "LPR" / "detections"
    test_imgs = (ROOT / "models" / "splits" / "lpd_test_split.txt").read_text().split()[: args.images]
    images = []
    for f in test_imgs:
        res = reader.predict(lpd / f)
        images.append({"file": f, "plates": [{"text": r.text, "box": [float(v) for v in r.box], "det_conf": r.det_conf,
                                             "text_conf": r.text_conf} for r in res]})
    crops_df = pd.read_csv(ROOT / "models" / "splits" / "lpr_test_split.csv", dtype=str).head(args.crops)
    crops = []
    for f, label in zip(crops_df.filename, crops_df.label):
        (text, conf, _), = reader.recognize([cv2.imread(str(lpr / f))])
        crops.append({"file": f, "label": label, "text": text, "text_conf": conf})
    pixels = []
    for f in crops_df.filename[:3]:
        src = R.to_pil(cv2.imread(str(lpr / f)), bgr=True)
        img = R.letterbox(src)
        pixels.append({"file": f, "src_w": src.size[0], "src_h": src.size[1],
                       "src_base64": base64.b64encode(np.asarray(src, dtype=np.uint8).tobytes()).decode(),
                       "rgb_base64": base64.b64encode(np.asarray(img, dtype=np.uint8).tobytes()).decode()})
    out = ROOT / "android" / "core" / "src" / "test" / "resources" / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {**reader.cfg, "yolo_imgsz": mob["imgsz"], "yolo_conf_threshold": mob["conf"]}
    out.write_text(json.dumps({"images": images, "crops": crops, "letterbox_pixels": pixels, "config": cfg,
                               "det_model": det_rel, "rec_model": rec_rel}, ensure_ascii=False, indent=1), encoding="utf-8")
    n_pl = sum(len(i["plates"]) for i in images)
    print(f"wrote {out}: {len(images)} images ({n_pl} plates), {len(crops)} crops, {len(pixels)} pixel references")


if __name__ == "__main__":
    main()
