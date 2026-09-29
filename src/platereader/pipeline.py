"""Phase 3: end-to-end licence-plate reading = YOLO26 detection -> padded crop -> CNN recognition.

Two interchangeable backends over the same model bundle (models/):
* ``torch`` (default): Ultralytics YOLO (lpd_best.pt) + TorchScript recognizer (lpr_best.ts).
* ``onnx``: ONNX Runtime for both models (lpd_best.onnx, lpr_best.onnx) with pre/post-processing written out in
  numpy. This is the reference the Android app mirrors, and it needs no PyTorch at inference time.

    from platereader.pipeline import PlateRecognizer
    reader = PlateRecognizer("models")
    for plate in reader.predict("car.jpg"):
        print(plate.text, plate.persian, plate.det_conf, plate.text_conf, plate.box)
"""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import recognizer as R
from .plates import to_latin, to_persian_display


@dataclass
class PlateResult:
    box: list                 # [x1, y1, x2, y2] in original-image pixels
    det_conf: float           # detector confidence
    text: str                 # plate string in label order, e.g. '12ب34567'
    text_conf: float          # min per-character probability (low = unreliable read)
    char_conf: list = field(default_factory=list)

    @property
    def latin(self):
        return to_latin(self.text)

    @property
    def persian(self):
        return to_persian_display(self.text)

    def to_dict(self):
        d = asdict(self)
        d.update(latin=self.latin, persian=self.persian)
        return d


def _to_bgr(image):
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(image)
        return img
    if hasattr(image, "convert"):  # PIL
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return np.asarray(image)


# ---------------------------------------------------------------- ONNX reference pre/post-processing (mirrored on Android)
def yolo_letterbox(img_bgr, size, pad=114):
    """Ultralytics LetterBox(auto=False): keep aspect, INTER_LINEAR resize, centre pad. Returns NCHW float RGB 0-1."""
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    img = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img_bgr
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))
    x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, left, top


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def yolo_postprocess(out, conf, iou, r, padx, pady, W, H, max_det=300):
    """Raw YOLO output [1, 5, N] (cx, cy, w, h, score) -> boxes in original pixels + scores (after NMS)."""
    p = out[0].T  # [N, 5]
    p = p[p[:, 4] >= conf]
    if not len(p):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    keep = nms(boxes, p[:, 4], iou)[:max_det]
    boxes, scores = boxes[keep], p[keep, 4]
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - padx) / r).clip(0, W)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pady) / r).clip(0, H)
    return boxes.astype(np.float32), scores.astype(np.float32)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class PlateRecognizer:
    """End-to-end reader. `models_dir` holds lpd_best.pt/.onnx, lpr_best.ts/.onnx and lpr_config.json."""

    def __init__(self, models_dir="models", backend="torch", device=None, conf=None, iou=0.7, max_plates=None,
                 det_model=None, rec_model=None, imgsz=None, threads=None):
        """det_model / rec_model / imgsz override the bundle defaults (e.g. INT8 or 640 px mobile models, ONNX only)."""
        self.dir = Path(models_dir)
        self.cfg = json.loads((self.dir / "lpr_config.json").read_text(encoding="utf-8"))
        self.letters = self.cfg["letter_vocab"]
        self.imgsz = int(imgsz or self.cfg["yolo_imgsz"])
        self.conf = float(self.cfg["yolo_conf_threshold"] if conf is None else conf)
        self.iou, self.max_plates, self.backend = iou, max_plates, backend
        self.crop_padding = float(self.cfg["crop_padding"])
        self.timings = {"detect_ms": [], "recognize_ms": []}
        if backend == "torch":
            import torch
            from ultralytics import YOLO

            from .utils import pick_device

            self.torch = torch
            self.device = pick_device(device)
            self.yolo_device = 0 if self.device.type == "cuda" else self.device.type
            self.detector = YOLO(str(self.dir / "lpd_best.pt"))
            self.recognizer = torch.jit.load(str(self.dir / "lpr_best.ts"), map_location=self.device).eval()
        elif backend == "onnx":
            import onnxruntime as ort

            prov = ["CPUExecutionProvider"]
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.det_sess = ort.InferenceSession(str(det_model or self.dir / "lpd_best.onnx"), so, providers=prov)
            self.rec_sess = ort.InferenceSession(str(rec_model or self.dir / "lpr_best.onnx"), so, providers=prov)
        else:
            raise ValueError(backend)

    # ------------------------------------------------------------ detection
    def detect(self, image, conf=None):
        """-> (boxes [N,4] xyxy pixels, scores [N]) sorted by score."""
        img = _to_bgr(image)
        conf = self.conf if conf is None else conf
        t0 = time.perf_counter()
        if self.backend == "torch":
            r = self.detector.predict(img, imgsz=self.imgsz, conf=conf, iou=self.iou, device=self.yolo_device, verbose=False)[0]
            boxes, scores = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        else:
            x, ratio, padx, pady = yolo_letterbox(img, self.imgsz)
            out = self.det_sess.run(None, {self.det_sess.get_inputs()[0].name: x})[0]
            boxes, scores = yolo_postprocess(out, conf, self.iou, ratio, padx, pady, img.shape[1], img.shape[0])
        self.timings["detect_ms"].append((time.perf_counter() - t0) * 1000)
        order = np.argsort(-scores)
        return boxes[order], scores[order]

    # ------------------------------------------------------------ recognition
    def recognize(self, crops_bgr):
        """List of BGR plate crops -> list of (text, text_conf, per-char probabilities)."""
        if not len(crops_bgr):
            return []
        t0 = time.perf_counter()
        x = np.stack([R.preprocess(c, bgr=True).numpy() for c in crops_bgr]).astype(np.float32)
        if self.backend == "torch":
            with self.torch.no_grad():
                d, l = self.recognizer(self.torch.from_numpy(x).to(self.device))
            d, l = d.float().cpu().numpy(), l.float().cpu().numpy()
        else:
            d, l = self.rec_sess.run(None, {"image": x})
        pd_, pl = softmax(d), softmax(l)
        out = []
        for i in range(len(crops_bgr)):
            di, li = pd_[i].argmax(-1), int(pl[i].argmax())
            digits = "".join(R.DIGIT_VOCAB[j] for j in di)
            text = digits[:2] + self.letters[li] + digits[2:]
            probs = list(pd_[i].max(-1)[:2]) + [pl[i].max()] + list(pd_[i].max(-1)[2:])
            out.append((text, float(min(probs)), [round(float(p), 4) for p in probs]))
        self.timings["recognize_ms"].append((time.perf_counter() - t0) * 1000)
        return out

    # ------------------------------------------------------------ end to end
    def predict(self, image, conf=None):
        img = _to_bgr(image)
        boxes, scores = self.detect(img, conf)
        if self.max_plates:
            boxes, scores = boxes[: self.max_plates], scores[: self.max_plates]
        crops, keep = [], []
        for i, b in enumerate(boxes):
            c = R.crop_with_padding(img, b, self.crop_padding)
            if c.size and c.shape[0] >= 2 and c.shape[1] >= 2:
                crops.append(c)
                keep.append(i)
        reads = self.recognize(crops)
        return [PlateResult([round(float(v), 1) for v in boxes[i]], round(float(scores[i]), 4), t, round(tc, 4), cc)
                for i, (t, tc, cc) in zip(keep, reads)]

    __call__ = predict

    def speed_summary(self):
        return {k: (float(np.mean(v)) if v else None) for k, v in self.timings.items()}

    def draw(self, image, results, font_path=None):
        """RGB numpy image with boxes and labels (Persian if a TTF font with Persian glyphs is available)."""
        from .viz import draw_plates

        return draw_plates(cv2.cvtColor(_to_bgr(image), cv2.COLOR_BGR2RGB), results, font_path=font_path)
