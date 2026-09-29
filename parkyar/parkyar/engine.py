"""Plate reader: YOLO26n detector + CNN recognizer (INT8 ONNX models), NumPy / ONNX Runtime only.

Pre- and post-processing are the same as the training project (`platereader.pipeline`, ONNX backend):
Ultralytics letterbox with OpenCV INTER_LINEAR for the detector, Pillow bilinear letterbox to 64x256 for the recognizer.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODELS = Path(__file__).parent / "assets" / "models"


@dataclass
class Plate:
    text: str
    conf: float  # lowest of the 8 character probabilities
    box: tuple  # x1, y1, x2, y2 in frame pixels
    det_score: float
    crop: np.ndarray  # BGR


def _yolo_letterbox(img, size, pad=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))
    x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, left, top


def _nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return np.array(keep, dtype=int)


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class PlateEngine:
    def __init__(self, models_dir=MODELS, threads=None):
        import onnxruntime as ort

        d = Path(models_dir)
        self.cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        self.letters = self.cfg["letters"]
        self.imgsz, self.conf, self.pad = int(self.cfg["det_imgsz"]), float(self.cfg["det_conf"]), float(self.cfg["crop_padding"])
        self.mean = np.array(self.cfg["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.array(self.cfg["std"], np.float32).reshape(3, 1, 1)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(d / "detector.onnx"), so, providers=prov)
        self.rec = ort.InferenceSession(str(d / "recognizer.onnx"), so, providers=prov)
        self.det_in = self.det.get_inputs()[0].name
        self.last_ms = 0.0

    # -------------------------------------------------------------- detection
    def detect(self, bgr, conf=None):
        conf = self.conf if conf is None else conf
        H, W = bgr.shape[:2]
        x, r, px, py = _yolo_letterbox(bgr, self.imgsz)
        p = self.det.run(None, {self.det_in: x})[0][0].T  # [N, 5]: cx, cy, w, h, score
        p = p[p[:, 4] >= conf]
        if not len(p):
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        b = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
        keep = _nms(b, p[:, 4], 0.7)
        b, s = b[keep], p[keep, 4]
        b[:, [0, 2]] = ((b[:, [0, 2]] - px) / r).clip(0, W)
        b[:, [1, 3]] = ((b[:, [1, 3]] - py) / r).clip(0, H)
        return b, s

    # -------------------------------------------------------------- recognition
    def _prep(self, crop_bgr):
        H, W = self.cfg["input_hw"]
        img = Image.fromarray(np.ascontiguousarray(crop_bgr[:, :, ::-1]))
        w, h = img.size
        s = min(W / w, H / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        canvas = Image.new("RGB", (W, H), (self.cfg["pad"],) * 3)
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((W - nw) // 2, (H - nh) // 2))
        x = np.asarray(canvas, np.float32).transpose(2, 0, 1) / 255.0
        return (x - self.mean) / self.std

    def recognize(self, crops):
        if not crops:
            return []
        d, l = self.rec.run(None, {"image": np.stack([self._prep(c) for c in crops]).astype(np.float32)})
        pd_, pl = _softmax(d), _softmax(l)
        out = []
        for i in range(len(crops)):
            digits = "".join(str(j) for j in pd_[i].argmax(-1))
            text = digits[:2] + self.letters[int(pl[i].argmax())] + digits[2:]
            out.append((text, float(min(pd_[i].max(-1).min(), pl[i].max()))))
        return out

    # -------------------------------------------------------------- end to end
    def read(self, bgr, conf=None, max_plates=5):
        t0 = time.perf_counter()
        boxes, scores = self.detect(bgr, conf)
        H, W = bgr.shape[:2]
        crops, kept = [], []
        for b, s in list(zip(boxes, scores))[:max_plates]:
            x1, y1, x2, y2 = [float(v) for v in b]
            bw, bh = x2 - x1, y2 - y1
            cx1, cy1 = max(0, int(round(x1 - self.pad * bw))), max(0, int(round(y1 - self.pad * bh)))
            cx2, cy2 = min(W, int(round(x2 + self.pad * bw))), min(H, int(round(y2 + self.pad * bh)))
            c = bgr[cy1:cy2, cx1:cx2]
            if c.shape[0] >= 2 and c.shape[1] >= 2:
                crops.append(c.copy())
                kept.append(((x1, y1, x2, y2), float(s)))
        reads = self.recognize(crops)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return [Plate(t, c, b, s, crop) for (b, s), (t, c), crop in zip(kept, reads, crops)]
