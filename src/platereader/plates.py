"""Iranian plate text: normalization, parsing, Latin transliteration and Persian display.

A plate label is 8 tokens in label order: 2 digits, 1 letter, 3 digits, 2-digit region code (e.g. ``12ب34567``).
The letter *alef* is stored spelled out (``الف``, 3 code points) but is ONE token.
"""
import re

# Letters in Persian alphabet order with a Latin transliteration (matplotlib/OpenCV cannot shape Persian script).
LETTER_LATIN = {
    "الف": "Alef", "آ": "A", "ب": "B", "پ": "P", "ت": "T", "ث": "Se", "ج": "J", "چ": "Ch", "ح": "He",
    "خ": "Kh", "د": "D", "ذ": "Zal", "ر": "R", "ز": "Z", "ژ": "Zh", "س": "S", "ش": "Sh", "ص": "Sad",
    "ض": "Zad", "ط": "Ta", "ظ": "Za", "ع": "Ein", "غ": "Ghein", "ف": "F", "ق": "Gh", "ک": "K", "گ": "G",
    "ل": "L", "م": "M", "ن": "N", "و": "V", "ه": "H", "ی": "Y",
}
DIGIT_VOCAB = list("0123456789")
PLATE_LEN, LETTER_POS, DIGIT_POS = 8, 2, [0, 1, 3, 4, 5, 6, 7]

_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_TO_PERSIAN = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_PLATE_RE = re.compile(r"^(\d{2})(\D+)(\d{5})$")


def normalize_plate(text):
    """Canonical plate string: ASCII digits, no ZWJ/ZWNJ/spaces/marks, Persian (not Arabic) yeh and kaf."""
    t = str(text).strip().translate(_TO_ASCII)
    for ch in ("‍", "‌", " ", "‏", "‎"):
        t = t.replace(ch, "")
    return t.replace("ي", "ی").replace("ك", "ک")


def parse_plate(text):
    """'12ب34567' -> ('1234567', 'ب');  None if it is not 2 digits + known letter + 5 digits."""
    m = _PLATE_RE.match(normalize_plate(text))
    if not m or m.group(2) not in LETTER_LATIN:
        return None
    return m.group(1) + m.group(3), m.group(2)


def plate_tokens(text):
    """8 tokens [d, d, letter, d, d, d, d, d] for a valid plate, else one token per character."""
    p = parse_plate(text)
    if p is None:
        return list(normalize_plate(text))
    d, letter = p
    return list(d[:2]) + [letter] + list(d[2:])


def to_latin(text):
    """'12ب34567' -> '12-B-345-67' (safe for plots and OpenCV)."""
    p = parse_plate(text)
    if p is None:
        return "?" + normalize_plate(text).encode("ascii", "replace").decode()
    d, letter = p
    return f"{d[:2]}-{LETTER_LATIN[letter]}-{d[2:5]}-{d[5:]}"


def to_persian_digits(text):
    return str(text).translate(_TO_PERSIAN)


def plate_parts(text):
    """Display parts of a plate: (two digits, letter, three digits, region) in Persian digits, or None."""
    p = parse_plate(text)
    if p is None:
        return None
    d, letter = p
    return to_persian_digits(d[:2]), letter, to_persian_digits(d[2:5]), to_persian_digits(d[5:])


def to_persian_display(text):
    """'12ب34567' -> '۱۲ ب ۳۴۵ - ۶۷' (as written on the plate, region code last)."""
    parts = plate_parts(text)
    if parts is None:
        return to_persian_digits(normalize_plate(text))
    a, letter, b, region = parts
    return f"{a} {letter} {b} - {region}"


def edit_distance(a, b):
    """Levenshtein distance between two token sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
