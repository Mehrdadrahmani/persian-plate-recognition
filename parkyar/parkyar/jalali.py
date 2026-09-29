"""Gregorian -> Jalali (Shamsi) conversion and Persian date / duration / money formatting."""
import datetime as dt

from .plates import fa_digits

MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
WEEKDAYS = ["دوشنبه", "سه‌شنبه", "چهارشنبه", "پنج‌شنبه", "جمعه", "شنبه", "یکشنبه"]  # indexed by date.weekday()


def to_jalali(y, m, d):
    """Gregorian date -> (jy, jm, jd). Arithmetic algorithm from jalaali-js (valid for 1600–3000 AD)."""
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    gy2 = y + 1 if m > 2 else y
    days = 355666 + 365 * y + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400 + d + g_d_m[m - 1]
    jy = -1595 + 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        return jy, 1 + days // 31, 1 + days % 31
    return jy, 7 + (days - 186) // 30, 1 + (days - 186) % 30


def _dt(ts):
    return ts if isinstance(ts, dt.datetime) else dt.datetime.fromtimestamp(ts)


def date(ts, weekday=False):
    """۵ مهر ۱۴۰۵ (optionally with the weekday)."""
    t = _dt(ts)
    jy, jm, jd = to_jalali(t.year, t.month, t.day)
    s = f"{fa_digits(jd)} {MONTHS[jm - 1]} {fa_digits(jy)}"
    return f"{WEEKDAYS[t.weekday()]} {s}" if weekday else s


def short_date(ts):
    """۱۴۰۵/۰۷/۰۵"""
    t = _dt(ts)
    jy, jm, jd = to_jalali(t.year, t.month, t.day)
    return fa_digits(f"{jy}/{jm:02d}/{jd:02d}")


def time(ts, seconds=False):
    return fa_digits(_dt(ts).strftime("%H:%M:%S" if seconds else "%H:%M"))


def datetime(ts):
    return f"{short_date(ts)}  {time(ts)}"


def duration(minutes):
    """۲ ساعت و ۱۵ دقیقه / ۱ روز و ۳ ساعت / ۴۰ دقیقه"""
    m = max(0, int(round(minutes)))
    d, rem = divmod(m, 1440)
    h, mm = divmod(rem, 60)
    items = [f"{fa_digits(d)} روز"] if d else []
    if h:
        items.append(f"{fa_digits(h)} ساعت")
    if mm and not d:
        items.append(f"{fa_digits(mm)} دقیقه")
    return " و ".join(items) or "کمتر از ۱ دقیقه"


def money(amount, unit=True):
    """۱۲٬۵۰۰ تومان"""
    s = fa_digits(f"{int(amount):,}".replace(",", "٬"))
    return f"{s} تومان" if unit else s
