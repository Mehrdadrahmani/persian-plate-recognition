"""Mobile optimization: INT8 quantization + a 640 px detector; chosen on the validation split, reported on the test split.

    python scripts/07_optimize_mobile.py            # -> models/mobile/*.onnx, reports/mobile/

Variants (all ONNX, scored with the Phase 3 end-to-end evaluation; the choice uses only the validation split):
  detector  : FP32 960 (baseline) | INT8 960 | INT8 960 with FP32 head | FP32 640 | INT8 640 (head FP32)
  recognizer: FP32 | INT8
INT8 = ONNX Runtime static quantization (QDQ, per-channel weights) calibrated on training images only.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from platereader import e2e_eval as E, recognizer as R  # noqa: E402
from platereader.config import add_common_args, config_from_args  # noqa: E402
from platereader.pipeline import PlateRecognizer, yolo_letterbox  # noqa: E402
from platereader.utils import RunLog  # noqa: E402


class _Reader:
    """onnxruntime CalibrationDataReader over a list of input arrays."""

    def __init__(self, name, arrays):
        self.name, self.it = name, iter(arrays)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {self.name: x}


def quantize(src, dst, input_name, arrays, exclude_prefix=None, log=print):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnx

    pre = Path(dst).with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    graph = onnx.load(str(pre)).graph.node
    # keep the detection head (prefix) and every Softmax in FP32: a quantized head breaks detection, and
    # XNNPACK (fast ARM backend on Android) has no kernel for quantized Softmax
    exclude = [n.name for n in graph if (exclude_prefix and n.name.startswith(exclude_prefix)) or n.op_type == "Softmax"]
    quantize_static(str(pre), str(dst), _Reader(input_name, arrays), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude)
    pre.unlink()
    log(f"quantized {Path(dst).name}: {Path(src).stat().st_size / 1e6:.1f} MB -> {Path(dst).stat().st_size / 1e6:.1f} MB"
        + (f" ({len(exclude)} nodes kept FP32)" if exclude else ""))


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--calib", type=int, default=128, help="calibration images per model (training split)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate on the first N validation/test images only")
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = cfg.paths
    out = P.reports("mobile")
    mob = P.models_dir / "mobile"
    mob.mkdir(parents=True, exist_ok=True)
    log = RunLog(out / "run_log.txt")
    rng = np.random.default_rng(cfg.seed)

    # ---------------- calibration data (training images only)
    train_imgs = (P.models_dir / "splits" / "lpd_train_split.txt").read_text().split()
    calib_imgs = [cv2.imread(str(P.lpd_dir / "images" / f)) for f in rng.choice(train_imgs, args.calib, replace=False)]
    train_crops = pd.read_csv(P.models_dir / "splits" / "lpr_train_split.csv", dtype=str).filename.tolist()
    calib_crops = [R.preprocess(cv2.imread(str(P.lpr_dir / "detections" / f)), bgr=True).numpy()[None]
                   for f in rng.choice(train_crops, 4 * args.calib, replace=False)]

    # ---------------- detector variants
    from ultralytics import YOLO

    det = {"fp32_960": (P.models_dir / "lpd_best.onnx", 960)}
    tmp = P.work_dir / "export640"  # export from a copy: Ultralytics writes <weights>.onnx next to the .pt
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(P.models_dir / "lpd_best.pt", tmp / "lpd_best.pt")
    p640 = Path(YOLO(str(tmp / "lpd_best.pt")).export(format="onnx", imgsz=640, device="cpu", verbose=False))
    shutil.move(str(p640), mob / "lpd_fp32_640.onnx")
    det["fp32_640"] = (mob / "lpd_fp32_640.onnx", 640)
    for name, (src, size, excl) in {"int8_960": (det["fp32_960"][0], 960, None),
                                    "int8h_960": (det["fp32_960"][0], 960, "/model.23/"),
                                    "int8h_640": (det["fp32_640"][0], 640, "/model.23/")}.items():
        arrays = [yolo_letterbox(im, size)[0] for im in calib_imgs]
        quantize(src, mob / f"lpd_{name}.onnx", "images", arrays, excl, log)
        det[name] = (mob / f"lpd_{name}.onnx", size)
    rec = {"fp32": P.models_dir / "lpr_best.onnx"}
    quantize(rec["fp32"], mob / "lpr_int8.onnx", "image", calib_crops, None, log)
    rec["int8"] = mob / "lpr_int8.onnx"

    # ---------------- recognizer alone on all LPR test crops
    test_crops = pd.read_csv(P.models_dir / "splits" / "lpr_test_split.csv", dtype=str)
    crops = [cv2.imread(str(P.lpr_dir / "detections" / f)) for f in test_crops.filename]
    rec_rows = []
    for rname, rpath in rec.items():
        r = PlateRecognizer(P.models_dir, backend="onnx", rec_model=rpath, threads=4)
        t0 = time.perf_counter()
        texts = [t for i in range(0, len(crops), 64) for t, _, _ in r.recognize(crops[i: i + 64])]
        ms = (time.perf_counter() - t0) * 1000 / len(crops)
        acc = float(np.mean([a == b for a, b in zip(texts, test_crops.label)]))
        rec_rows.append({"recognizer": rname, "test_plate_acc": acc, "ms_per_crop_batched": ms, "size_MB": rpath.stat().st_size / 1e6})
        log(f"recognizer {rname}: test full-plate accuracy {acc:.4f}, {ms:.2f} ms/crop, {rpath.stat().st_size / 1e6:.1f} MB")
    pd.DataFrame(rec_rows).to_csv(out / "recognizer_variants.csv", index=False)

    # ---------------- end-to-end for each combination: choose on VALIDATION, report on TEST
    def split_items(name):
        files = (P.models_dir / "splits" / f"lpd_{name}_split.txt").read_text().split()[: args.limit]
        return E.load_ground_truth(P.lpd_dir, files)

    val_items, test_items = split_items("val"), split_items("test")
    combos = [("fp32_960", "fp32"), ("int8_960", "fp32"), ("int8h_960", "fp32"), ("fp32_640", "fp32"),
              ("int8h_640", "fp32"), ("int8h_960", "int8"), ("int8h_640", "int8")]
    rows = []
    for dname, rname in combos:
        dpath, size = det[dname]
        reader = PlateRecognizer(P.models_dir, backend="onnx", det_model=dpath, rec_model=rec[rname], imgsz=size, threads=4)
        vcache, _ = E.run_pipeline(reader, val_items, log=lambda *a: None)
        vsweep = E.threshold_sweep(val_items, vcache)
        vbest = vsweep.loc[vsweep.e2e_f1.idxmax()]
        thr = float(vbest.threshold)
        tcache, timing = E.run_pipeline(reader, test_items, log=lambda *a: None)
        t = E.evaluate_at(test_items, tcache, thr)[0]
        row = {"detector": dname, "recognizer": rname, "det_imgsz": size, "threshold_from_val": thr,
               "val_e2e_f1": float(vbest.e2e_f1), "test_e2e_f1": t["e2e_f1"], "test_e2e_precision": t["e2e_precision"],
               "test_e2e_recall": t["e2e_recall"], "test_det_recall": t["det_recall"], "test_correct_reads": t["correct_reads"],
               "test_gt_plates": t["gt_plates"], "test_cer": t["cer_on_detected"],
               "det_MB": dpath.stat().st_size / 1e6, "rec_MB": rec[rname].stat().st_size / 1e6,
               "detect_ms": timing["detect_ms"], "recognize_ms": timing["recognize_ms"], "total_ms": timing["total_ms"]}
        rows.append(row)
        log(f"{dname:9s} + rec {rname:4s}: VAL F1 {row['val_e2e_f1']:.4f} @ {thr:.2f} | TEST F1 {row['test_e2e_f1']:.4f} "
            f"({row['test_correct_reads']}/{row['test_gt_plates']}), det recall {row['test_det_recall']:.4f} | "
            f"{row['total_ms']:.0f} ms/img (det {row['detect_ms']:.0f}, rec {row['recognize_ms']:.0f}) | "
            f"{row['det_MB'] + row['rec_MB']:.0f} MB")
    table = pd.DataFrame(rows)
    table.to_csv(out / "mobile_variants.csv", index=False)

    # ---------------- choice (validation only): best val F1; ties within 0.2 points -> the faster one
    top = table.val_e2e_f1.max()
    pick = table[table.val_e2e_f1 >= top - 0.002].sort_values("total_ms").iloc[0]
    shutil.copy2(det[pick.detector][0], mob / "lpd_mobile.onnx")
    shutil.copy2(rec[pick.recognizer], mob / "lpr_mobile.onnx")
    base = table.iloc[0]
    choice = {"detector": pick.detector, "recognizer": pick.recognizer, "imgsz": int(pick.det_imgsz),
              "conf": float(pick.threshold_from_val), "val_e2e_f1": float(pick.val_e2e_f1),
              "test_e2e_f1": float(pick.test_e2e_f1), "test_correct_reads": int(pick.test_correct_reads),
              "test_gt_plates": int(pick.test_gt_plates), "size_MB": float(pick.det_MB + pick.rec_MB),
              "ms_per_image_mac_cpu": float(pick.total_ms),
              "baseline": {"test_e2e_f1": float(base.test_e2e_f1), "size_MB": float(base.det_MB + base.rec_MB),
                           "ms_per_image_mac_cpu": float(base.total_ms)}}
    (mob / "mobile_config.json").write_text(json.dumps(choice, indent=2))
    (out / "mobile_choice.json").write_text(json.dumps(choice, indent=2))
    for f in mob.glob("lpd_*.onnx"):  # keep only the shipped files
        if f.name != "lpd_mobile.onnx":
            f.unlink()
    (mob / "lpr_int8.onnx").unlink(missing_ok=True)
    log("chosen (on validation):", json.dumps(choice))


if __name__ == "__main__":
    import os

    main()
    sys.stdout.flush()
    os._exit(0)  # onnxruntime + torch can segfault during interpreter teardown; all results are written by now
