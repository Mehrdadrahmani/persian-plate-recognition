"""Drawing helpers: plate boxes with Persian text (correctly shaped, right-to-left) on images."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
DEFAULT_FONT = FONT_DIR / "Vazirmatn-Bold.ttf"


def shape_persian(text):
    """Reshape + bidi-reorder Persian text so PIL draws it correctly (falls back to the raw text)."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def plate_label(result, index):
    """Label in the plate's own visual order (left to right, as printed on the plate): '۱:  ۴۱ ط ۷۶۲  ۸۸   ۷۸٪'.
    Only the letter is shaped/reordered (matters for the 3-letter 'الف'); PIL draws the rest left to right."""
    from .plates import plate_parts, to_persian_digits

    conf = to_persian_digits(f"{result.text_conf * 100:.0f}") + "٪"
    parts = plate_parts(result.text)
    if parts is None:
        return f"{to_persian_digits(index)}:  {shape_persian(result.text)}   {conf}"
    a, letter, b, region = parts
    return f"{to_persian_digits(index)}:  {a} {shape_persian(letter)} {b}  {region}   {conf}"


def _font(size, font_path=None):
    p = Path(font_path) if font_path else DEFAULT_FONT
    try:
        return ImageFont.truetype(str(p), size)
    except OSError:
        return ImageFont.load_default()


def draw_plates(img_rgb, results, font_path=None, color=(229, 57, 53)):
    """Draw each result's box and a label (Persian plate + confidence) above it. Returns an RGB numpy array."""
    im = Image.fromarray(np.asarray(img_rgb)).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, W // 300)
    font = _font(max(16, W // 40), font_path)
    for i, r in enumerate(results, 1):
        x1, y1, x2, y2 = r.box
        d.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = plate_label(r, i)
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 3 * lw if y1 - th - 3 * lw > 0 else y2 + lw
        tx = min(max(0, x1), max(0, W - tw - 2 * lw))
        d.rectangle([tx, ty, tx + tw + 2 * lw, ty + th + 2 * lw], fill=color)
        d.text((tx + lw - tb[0], ty + lw - tb[1]), label, font=font, fill=(255, 255, 255))
    return np.asarray(im)
