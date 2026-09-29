"""Iranian plate strings: 2 digits + letter + 3 digits + 2-digit region code, e.g. "12ب34567"."""
import re

_FA = "۰۱۲۳۴۵۶۷۸۹"
_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_TO_FA = str.maketrans("0123456789", _FA)
_PLATE = re.compile(r"^(\d{2})(\D+)(\d{3})(\d{2})$")


def normalize(text):
    """ASCII digits, no spaces / joiners / separators, Persian yeh and kaf."""
    t = str(text).strip().translate(_TO_ASCII)
    for ch in ("‍", "‌", " ", "‏", "‎", "-", "|"):
        t = t.replace(ch, "")
    return t.replace("ي", "ی").replace("ك", "ک")


def parts(text):
    """-> (first2, letter, middle3, region2) or None if the text is not a valid plate."""
    m = _PLATE.match(normalize(text))
    return m.groups() if m else None


def is_valid(text):
    return parts(text) is not None


def fa_digits(s):
    return str(s).translate(_TO_FA)


def display(text):
    """Human-readable Persian form: ۱۲ ب ۳۴۵ - ۶۷"""
    p = parts(text)
    if not p:
        return fa_digits(text)
    return f"{fa_digits(p[0])} {p[1]} {fa_digits(p[2])} - {fa_digits(p[3])}"


def display_parts(text):
    """Pieces in the plate's printed left-to-right order, for drawing one by one (bidi would reorder a string)."""
    p = parts(text)
    if not p:
        return [fa_digits(text)]
    return [fa_digits(p[0]), p[1], fa_digits(p[2]), "|", fa_digits(p[3])]


def latin(text, letters, letters_latin):
    """ASCII form for CSV / search: 12-B-345-67"""
    p = parts(text)
    if not p:
        return text
    lat = dict(zip(letters, letters_latin)).get(p[1], "?")
    return f"{p[0]}-{lat}-{p[2]}-{p[3]}"


def edit_distance(a, b):
    """Levenshtein distance on plate characters (the letter counts as one token)."""
    ta, tb = _tokens(a), _tokens(b)
    prev = list(range(len(tb) + 1))
    for i, ca in enumerate(ta, 1):
        cur = [i]
        for j, cb in enumerate(tb, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _tokens(text):
    p = parts(text)
    return list(p[0]) + [p[1]] + list(p[2] + p[3]) if p else list(normalize(text))
