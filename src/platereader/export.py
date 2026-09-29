"""Build the inference bundle (models/): detector weights + ONNX, best recognizer as TorchScript + ONNX, config."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import recognizer as R
from .plates import LETTER_LATIN, LETTER_POS, PLATE_LEN


def export_recognizer(ckpt_path, out_dir, sample=None, log=print):
    """TorchScript (traced on CPU) + ONNX of a recognizer checkpoint, both checked against the eager model."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = R.build_model(ck["arch"], ck["n_letters"], pretrained=False, backbone=ck.get("backbone") or "mobilenet_v3_large")
    model.load_state_dict(ck["state_dict"])
    model.eval()
    x = sample if sample is not None else torch.randn(4, 3, *R.INPUT_SIZE)
    with torch.no_grad():
        ts = torch.jit.trace(model, torch.zeros(1, 3, *R.INPUT_SIZE))
        ts.save(str(Path(out_dir) / "lpr_best.ts"))
        eager, traced = model(x), torch.jit.load(str(Path(out_dir) / "lpr_best.ts"))(x)
    diff = max((a - b).abs().max().item() for a, b in zip(eager, traced))
    assert diff < 1e-4, diff
    log(f"lpr_best.ts OK (max |eager - traced| = {diff:.1e})")
    torch.onnx.export(model, torch.zeros(1, 3, *R.INPUT_SIZE), str(Path(out_dir) / "lpr_best.onnx"), input_names=["image"],
                      output_names=["digit_logits", "letter_logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "batch"}, "digit_logits": {0: "batch"}, "letter_logits": {0: "batch"}})
    import onnxruntime as ort

    o = ort.InferenceSession(str(Path(out_dir) / "lpr_best.onnx"), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    diff = max(np.abs(a - b.numpy()).max() for a, b in zip(o, eager))
    assert diff < 1e-3, diff
    log(f"lpr_best.onnx OK (max |onnx - torch| = {diff:.1e})")
    return ck


def export_detector(best_pt, out_dir, imgsz, log=print):
    from ultralytics import YOLO

    shutil.copy2(best_pt, Path(out_dir) / "lpd_best.pt")
    p = Path(YOLO(str(Path(out_dir) / "lpd_best.pt")).export(format="onnx", imgsz=imgsz, device="cpu", verbose=False))
    if p.resolve() != (Path(out_dir) / "lpd_best.onnx").resolve():
        shutil.move(str(p), Path(out_dir) / "lpd_best.onnx")
    log("lpd_best.onnx exported")


def write_config(out_dir, ck, arch, imgsz, conf, crop_padding=0.05):
    vocab = ck["letter_vocab"]
    cfg = {
        "best_model": arch, "arch": arch, "backbone": ck.get("backbone") if arch == "pretrained" else None,
        "torchscript": "lpr_best.ts", "onnx": "lpr_best.onnx", "state_dict": f"lpr_{arch}_best.pt",
        "input_size_hw": list(R.INPUT_SIZE), "resize": "letterbox (keep aspect ratio, centre, pad with gray)",
        "pad_value": R.PAD_VALUE, "color_mode": "RGB", "normalization": {"mean": list(R.MEAN), "std": list(R.STD)},
        "digit_vocab": R.DIGIT_VOCAB, "letter_vocab": vocab, "letter_vocab_latin": [LETTER_LATIN[c] for c in vocab],
        "plate_len": PLATE_LEN, "position_layout": ["letter" if i == LETTER_POS else "digit" for i in range(PLATE_LEN)],
        "letter_position": LETTER_POS, "digit_positions": [0, 1, 3, 4, 5, 6, 7],
        "output_format": "DD L DDD DD without separators, e.g. '12ب34567' (last two digits = region code; alef is 'الف')",
        "crop_padding": crop_padding, "yolo_conf_threshold": round(float(conf), 3), "yolo_imgsz": int(imgsz),
    }
    (Path(out_dir) / "lpr_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg
