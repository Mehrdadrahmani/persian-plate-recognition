"""Gate camera worker: reads a webcam / IP camera / video / image, reads plates, confirms each car once."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from . import plates
from .widgets import bgr_to_qimage

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path):
    """cv2.imread that also works with non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def open_capture(source):
    s = str(source).strip()
    if s.isdigit():
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(s), backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap
    return cv2.VideoCapture(s)


class Tracker:
    """A plate is confirmed when the same text is read in `hits` frames within `window` seconds.
    A confirmed plate (or one that differs by one character) is ignored for `cooldown` seconds."""

    def __init__(self, hits=2, window=3.0, cooldown=60.0, min_conf=0.5):
        self.hits, self.window, self.cooldown, self.min_conf = hits, window, cooldown, min_conf
        self.seen = {}  # text -> list of (time, Plate)
        self.done = {}  # text -> confirm time

    def update(self, found, now=None):
        now = now or time.time()
        out = []
        for p in found:
            if p.conf < self.min_conf or not plates.is_valid(p.text):
                continue
            if any(now - t < self.cooldown and plates.edit_distance(p.text, d) <= 1 for d, t in self.done.items()):
                continue
            lst = [x for x in self.seen.get(p.text, []) if now - x[0] <= self.window] + [(now, p)]
            self.seen[p.text] = lst
            if len(lst) >= self.hits:
                out.append(max(lst, key=lambda x: x[1].conf)[1])
                self.done[p.text] = now
                self.seen.pop(p.text, None)
        self.done = {d: t for d, t in self.done.items() if now - t < self.cooldown}
        return out


class GateWorker(QThread):
    frame = Signal(object, list)  # QImage (display size), plates (boxes scaled to it)
    confirmed = Signal(object, object)  # Plate, full BGR frame
    failed = Signal(str)

    def __init__(self, source, engine, conf_fn, cooldown=60.0, parent=None):
        super().__init__(parent)
        self.source, self.engine, self.conf_fn = str(source), engine, conf_fn
        self.tracker = Tracker(cooldown=cooldown)
        self._stop = False

    def stop(self):
        self._stop = True
        self.wait(3000)

    def run(self):
        s = self.source
        still = Path(s).suffix.lower() in IMAGE_EXT
        cap, img, fps = None, None, 0
        if still:
            img = read_image(s)
            if img is None:
                self.failed.emit("تصویر باز نشد")
                return
        else:
            cap = open_capture(s)
            if not cap.isOpened():
                self.failed.emit("اتصال به دوربین برقرار نشد")
                return
            is_file = not s.isdigit() and "://" not in s
            fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
        misses = 0
        while not self._stop:
            t0 = time.time()
            if still:
                frame = img
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if fps:  # video file: loop it (handy for demos)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    misses += 1
                    if misses > 50:
                        self.failed.emit("تصویر دوربین قطع شد")
                        break
                    time.sleep(0.05)
                    continue
                misses = 0
            found = self.engine.read(frame, self.conf_fn())
            for p in self.tracker.update(found):
                self.confirmed.emit(p, frame)
            h, w = frame.shape[:2]
            k = min(1.0, 1280 / w)
            disp = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame
            shown = [type(p)(p.text, p.conf, tuple(v * k for v in p.box), p.det_score, None) for p in found]
            self.frame.emit(bgr_to_qimage(disp), shown)
            dt = time.time() - t0
            if still:
                time.sleep(max(0.0, 0.4 - dt))
            elif fps:
                time.sleep(max(0.0, 1.0 / fps - dt))
        if cap is not None:
            cap.release()
