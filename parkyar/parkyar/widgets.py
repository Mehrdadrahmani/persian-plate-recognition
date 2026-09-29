"""Reusable widgets: Iranian plate card, plate input, cards, KPI tiles, bar chart, camera view."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

from . import plates, theme
from .theme import font


def bgr_to_qimage(bgr):
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def button(text, icon_name=None, kind=None, color=None):
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if icon_name:
        c = color or ("#FFFFFF" if kind in ("primary", "success") else theme.RED if kind == "danger" else
                      theme.MUTED if kind == "ghost" else theme.BLUE)
        b.setIcon(theme.icon(icon_name, c, 20))
        b.setIconSize(QSize(20, 20))
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def label(text="", obj=None, size=None, weight=None, color=None, wrap=False):
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    if size or weight:
        lb.setFont(font(size or 10, weight or QFont.Weight.Normal))
    if color:
        lb.setStyleSheet(f"color: {color};")
    lb.setWordWrap(wrap)
    return lb


class Card(QFrame):
    def __init__(self, title=None, parent=None, padding=20):
        super().__init__(parent)
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(padding, padding - 2, padding, padding)
        self.lay.setSpacing(12)
        self.header = QHBoxLayout()
        if title is not None:
            self.title = label(title, "cardTitle")
            self.header.addWidget(self.title)
            self.header.addStretch(1)
            self.lay.addLayout(self.header)


class Kpi(Card):
    """Dashboard tile: icon badge, big value, caption, optional progress bar."""

    def __init__(self, icon_name, caption, color=theme.BLUE, soft=theme.BLUE_SOFT):
        super().__init__(padding=18)
        row = QHBoxLayout()
        badge = QLabel()
        badge.setFixedSize(48, 48)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"background: {soft}; border-radius: 14px;")
        badge.setPixmap(theme.pixmap(icon_name, color, 26))
        col = QVBoxLayout()
        col.setSpacing(0)
        self.value = label("—", "kpiValue")
        self.caption = label(caption, "kpiLabel")
        col.addWidget(self.value)
        col.addWidget(self.caption)
        row.addLayout(col, 1)
        row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        self.lay.addLayout(row)


class PlateWidget(QWidget):
    """Iranian licence plate drawn to scale (the plate layout is always left-to-right)."""

    def __init__(self, text="", height=64, parent=None):
        super().__init__(parent)
        self.text = text
        self.h = height
        self.setFixedSize(int(height * 4.4), height)
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

    def set_plate(self, text):
        self.text = text or ""
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        W, H = self.width(), self.height()
        r = QRectF(1, 1, W - 2, H - 2)
        rad = H * 0.14
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#111111"))
        p.drawRoundedRect(r, rad, rad)
        inner = r.adjusted(H * 0.06, H * 0.06, -H * 0.06, -H * 0.06)
        p.setBrush(QColor("white"))
        p.drawRoundedRect(inner, rad * 0.6, rad * 0.6)
        # blue strip with flag
        strip = QRectF(inner.left(), inner.top(), inner.height() * 0.42, inner.height())
        path = QPainterPath()
        path.addRoundedRect(strip, rad * 0.6, rad * 0.6)
        p.setBrush(QColor("#1F4FB5"))
        p.drawPath(path)
        fw, fh = strip.width() * 0.62, strip.height() * 0.24
        fx, fy = strip.center().x() - fw / 2, strip.top() + strip.height() * 0.14
        for i, c in enumerate(("#239F40", "#FFFFFF", "#DA0000")):
            p.setBrush(QColor(c))
            p.drawRect(QRectF(fx, fy + i * fh / 3, fw, fh / 3 + 0.5))
        p.setPen(QColor("white"))
        f = QFont("Arial")
        show_iran = H >= 44
        f.setPixelSize(max(6, int(H * 0.13)))
        f.setBold(True)
        p.setFont(f)
        if show_iran:
            p.drawText(QRectF(strip.left(), strip.top() + strip.height() * 0.46, strip.width(), strip.height() * 0.5),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "I.R.\nIRAN")
        # region box
        reg_w = inner.height() * 0.95
        reg = QRectF(inner.right() - reg_w, inner.top(), reg_w, inner.height())
        p.setPen(QPen(QColor("#111111"), max(1.5, H * 0.04)))
        p.drawLine(QPointF(reg.left(), inner.top()), QPointF(reg.left(), inner.bottom()))
        pt = plates.parts(self.text)
        p.setPen(QColor("#111111"))
        p.setFont(font(max(5, H * 0.13), QFont.Weight.Bold))
        if show_iran:
            p.drawText(QRectF(reg.left(), reg.top() + H * 0.02, reg.width(), reg.height() * 0.32), Qt.AlignmentFlag.AlignCenter, "ایران")
        big = font(1, QFont.Weight.Black)
        big.setPixelSize(int(H * 0.52))
        p.setFont(big)
        if pt:
            a, letter, b, region = pt
            top = 0.26 if show_iran else 0.0
            p.drawText(QRectF(reg.left(), reg.top() + reg.height() * top, reg.width(), reg.height() * (1 - top)),
                       Qt.AlignmentFlag.AlignCenter, plates.fa_digits(region))
            body = QRectF(strip.right(), inner.top(), reg.left() - strip.right(), inner.height())
            cells = [(plates.fa_digits(a), 0.28), (letter, 0.26), (plates.fa_digits(b), 0.46)]
            x = body.left()
            for s, share in cells:
                w = body.width() * share
                p.drawText(QRectF(x, body.top(), w, body.height()), Qt.AlignmentFlag.AlignCenter, s)
                x += w
        else:
            body = QRectF(strip.right(), inner.top(), reg.left() - strip.right(), inner.height())
            p.setPen(QColor(theme.MUTED))
            p.drawText(body, Qt.AlignmentFlag.AlignCenter, plates.fa_digits(self.text) if self.text else "—")
        p.end()


class PlateInput(QWidget):
    """Plate entry laid out like the plate: [2 digits] [letter] [3 digits] | [region]. Accepts Persian digits."""

    changed = Signal()

    def __init__(self, letters, parent=None):
        super().__init__(parent)
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        def field(n, ph):
            e = QLineEdit()
            e.setMaxLength(n)
            e.setPlaceholderText(ph)
            e.setAlignment(Qt.AlignmentFlag.AlignCenter)
            e.setFont(font(16, QFont.Weight.Bold))
            e.setFixedWidth(34 + 22 * n)
            e.textEdited.connect(lambda t, e=e, n=n: self._digits(e, n))
            return e

        self.a = field(2, "۱۲")
        self.letter = QComboBox()
        self.letter.addItems(letters)
        self.letter.setFont(font(15, QFont.Weight.Bold))
        self.letter.setFixedWidth(96)
        self.b = field(3, "۳۴۵")
        self.region = field(2, "۶۷")
        sep = label("ایران", color=theme.MUTED)
        for w in (self.a, self.letter, self.b, sep, self.region):
            lay.addWidget(w)
        lay.addStretch(1)
        self.letter.currentIndexChanged.connect(self.changed)

    def _digits(self, e, n):
        t = "".join(ch for ch in plates.normalize(e.text()) if ch.isdigit())[:n]
        e.setText(plates.fa_digits(t))
        self.changed.emit()
        if len(t) == n:
            nxt = {self.a: self.letter, self.b: self.region}.get(e)
            if nxt:
                nxt.setFocus()

    def text(self):
        return plates.normalize(self.a.text()) + self.letter.currentText() + plates.normalize(self.b.text() + self.region.text())

    def set_text(self, t):
        pt = plates.parts(t)
        if not pt:
            return
        self.a.setText(plates.fa_digits(pt[0]))
        i = self.letter.findText(pt[1])
        if i >= 0:
            self.letter.setCurrentIndex(i)
        self.b.setText(plates.fa_digits(pt[2]))
        self.region.setText(plates.fa_digits(pt[3]))

    def valid(self):
        return plates.is_valid(self.text())


class BarChart(QWidget):
    """Simple rounded bar chart (values with labels), drawn with QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, items, fmt=str):
        self.items, self.fmt = items, fmt
        self.update()

    def paintEvent(self, _):
        if not self.items:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        top, bottom = 26, 34
        n = len(self.items)
        vmax = max(v for _, v in self.items) or 1
        slot = W / n
        bw = min(46, slot * 0.5)
        for i, (lab, v) in enumerate(self.items):
            # right-to-left: the first (oldest) item is drawn on the right
            cx = W - slot * (i + 0.5)
            h = (H - top - bottom) * v / vmax
            last = i == n - 1
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#EDF1FA"))
            p.drawRoundedRect(QRectF(cx - bw / 2, top, bw, H - top - bottom), 8, 8)
            p.setBrush(QColor(theme.BLUE if last else "#9DB4F0"))
            if h > 0:
                p.drawRoundedRect(QRectF(cx - bw / 2, H - bottom - h, bw, h), 8, 8)
            p.setPen(QColor(theme.MUTED))
            p.setFont(font(8.5))
            p.drawText(QRectF(cx - slot / 2, H - bottom + 6, slot, 22), Qt.AlignmentFlag.AlignCenter, lab)
            if v:
                p.setPen(QColor(theme.TEXT))
                p.setFont(font(8, QFont.Weight.Bold))
                p.drawText(QRectF(cx - slot / 2, max(0, H - bottom - h - 22), slot, 20), Qt.AlignmentFlag.AlignCenter, self.fmt(v))
        p.end()


class CameraView(QWidget):
    """Video frame (letterboxed, rounded) with detected plates boxed on top."""

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.img = None
        self.plates = []
        self.placeholder = placeholder
        self.setMinimumSize(360, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, qimg, plates_=None):
        self.img = qimg
        if plates_ is not None:
            self.plates = plates_
        self.update()

    def set_plates(self, plates_):
        self.plates = plates_
        self.update()

    def clear(self):
        self.img, self.plates = None, []
        self.update()

    def heightForWidth(self, w):
        return int(w * 9 / 16)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(r, 14, 14)
        p.setClipPath(clip)
        p.fillRect(r, QColor("#0B1226"))
        if self.img is None:
            p.setPen(QColor("#7E8BAE"))
            p.setFont(font(11))
            p.drawPixmap(int(r.center().x() - 24), int(r.center().y() - 44), theme.pixmap("gate", "#3A4A78", 48))
            p.drawText(r.adjusted(20, 40, -20, 0), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            p.end()
            return
        iw, ih = self.img.width(), self.img.height()
        s = min(r.width() / iw, r.height() / ih)
        dw, dh = iw * s, ih * s
        ox, oy = (r.width() - dw) / 2, (r.height() - dh) / 2
        p.drawImage(QRectF(ox, oy, dw, dh), self.img)
        for pl in self.plates:
            x1, y1, x2, y2 = pl.box
            box = QRectF(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s)
            p.setPen(QPen(QColor(theme.AMBER), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(box, 4, 4)
            pieces = plates.display_parts(pl.text)
            p.setFont(font(11, QFont.Weight.Bold))
            fm = p.fontMetrics()
            gap = fm.horizontalAdvance(" ")
            widths = [fm.horizontalAdvance(s) for s in pieces]
            tw, th = sum(widths) + gap * (len(pieces) - 1) + 18, fm.height() + 6
            tag = QRectF(box.center().x() - tw / 2, box.top() - th - 6, tw, th)
            if tag.top() < 0:
                tag.moveTop(box.bottom() + 6)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.AMBER))
            p.drawRoundedRect(tag, 7, 7)
            p.setPen(QColor(theme.NAVY))
            x = tag.left() + 9
            for s, w in zip(pieces, widths):  # left to right, as printed on the plate
                p.drawText(QRectF(x, tag.top(), w, tag.height()), Qt.AlignmentFlag.AlignCenter, s)
                x += w + gap
        p.end()


def crop_pixmap(path_or_bgr, h=44, max_w=None):
    """Plate snapshot (file path or BGR array) as a pixmap of height h (and at most max_w wide)."""
    if path_or_bgr is None:
        return None
    if isinstance(path_or_bgr, str):
        pm = QPixmap(path_or_bgr)
        if pm.isNull():
            return None
    else:
        pm = QPixmap.fromImage(bgr_to_qimage(path_or_bgr))
    if max_w:
        return pm.scaled(QSize(max_w, h), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return pm.scaledToHeight(h, Qt.TransformationMode.SmoothTransformation)


def hline():
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {theme.LINE};")
    return f
