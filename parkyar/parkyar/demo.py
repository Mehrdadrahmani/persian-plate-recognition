"""Sample data for `parkyar --demo`: a week of visits, cars currently inside, and a few subscribers."""
import random
import time

from . import billing

LETTERS = ["ب", "د", "س", "ص", "ط", "ق", "ل", "م", "ن", "و", "ه", "ی", "ج", "ع"]


def _plate(rng):
    return f"{rng.randint(10, 99)}{rng.choice(LETTERS)}{rng.randint(100, 999)}{rng.randint(10, 99)}"


def seed(db, days=7, seed=1405):
    """Fill an empty database (no-op if it already has sessions)."""
    if db.con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]:
        return
    rng = random.Random(seed)
    t = billing.Tariff.from_settings(db.settings())
    now = time.time()
    lt = time.localtime(now)
    today = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    regulars = [_plate(rng) for _ in range(25)]
    rows = []
    for d in range(days, -1, -1):
        n = rng.randint(55, 95) if d else rng.randint(40, 60)
        for _ in range(n):
            start = today - d * 86400 + rng.uniform(7 * 3600, 21 * 3600)
            if start > now - 600:
                continue
            minutes = min(rng.lognormvariate(4.3, 0.8), 14 * 60)
            end = start + minutes * 60
            plate = rng.choice(regulars) if rng.random() < 0.2 else _plate(rng)
            if end > now:
                rows.append((plate, start, None, None, 0))
            else:
                rows.append((plate, start, end, billing.fee(minutes, t), 1))
    for _ in range(rng.randint(34, 42)):  # cars parked right now
        start = now - rng.uniform(5, 9 * 60) * 60
        rows.append((rng.choice(regulars) if rng.random() < 0.15 else _plate(rng), start, None, None, 0))
    # keep at most one open session per plate
    seen = set()
    with db.con:
        for plate, start, end, fee, paid in sorted(rows, key=lambda r: r[1], reverse=True):
            if end is None:
                if plate in seen:
                    continue
                seen.add(plate)
            db.con.execute("INSERT INTO sessions(plate, entry_time, exit_time, fee, paid) VALUES (?, ?, ?, ?, ?)",
                           (plate, start, end, fee, paid))
        names = ["رضا احمدی", "مریم کریمی", "علی رضایی", "سارا محمدی", "حسین موسوی", "نگار حسینی"]
        for i, name in enumerate(names):
            db.con.execute("INSERT OR IGNORE INTO subscribers(plate, name, phone, valid_until, created) VALUES (?, ?, ?, ?, ?)",
                           (regulars[i], name, f"0912{rng.randint(1000000, 9999999)}",
                            now + (rng.choice([3, 25, 60, 90]) if i != 4 else -5) * 86400, now - i * 3600))
        db.con.execute("UPDATE sessions SET subscriber = 1, fee = 0 WHERE plate IN (SELECT plate FROM subscribers)")
