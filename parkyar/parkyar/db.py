"""SQLite storage: parking sessions (entry -> exit), subscribers, settings. Times are Unix seconds (local clock)."""
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import plates

DEFAULTS = {
    "lot_name": "پارکینگ مرکزی",
    "capacity": 120,
    "free_minutes": 10,
    "first_hour": 30000,
    "extra_hour": 20000,
    "daily_cap": 250000,
    "cooldown_s": 60,
    "sensitivity": 50,
    "entry_source": "0",
    "exit_source": "1",
    "auto_entry": True,
    "auto_exit": True,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    entry_time REAL NOT NULL,
    exit_time REAL,
    fee INTEGER,
    paid INTEGER NOT NULL DEFAULT 0,
    subscriber INTEGER NOT NULL DEFAULT 0,
    entry_img TEXT,
    exit_img TEXT,
    manual INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_open ON sessions(exit_time);
CREATE INDEX IF NOT EXISTS idx_sessions_plate ON sessions(plate);
CREATE TABLE IF NOT EXISTS subscribers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL DEFAULT '',
    phone TEXT NOT NULL DEFAULT '',
    valid_until REAL NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass
class Session:
    id: int
    plate: str
    entry_time: float
    exit_time: float | None
    fee: int | None
    paid: bool
    subscriber: bool
    entry_img: str | None
    exit_img: str | None
    manual: bool

    @property
    def minutes(self):
        return ((self.exit_time or time.time()) - self.entry_time) / 60


@dataclass
class Subscriber:
    id: int
    plate: str
    name: str
    phone: str
    valid_until: float
    created: float

    @property
    def active(self):
        return self.valid_until >= time.time()


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.images = self.path.parent / "images"
        self.images.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self.con = sqlite3.connect(str(self.path), check_same_thread=False)
        self.con.executescript(SCHEMA)
        self.con.commit()

    # ------------------------------------------------------------------ settings
    def settings(self):
        with self._lock:
            s = dict(DEFAULTS)
            s.update({k: json.loads(v) for k, v in self.con.execute("SELECT key, value FROM settings")})
            return s

    def save_settings(self, values):
        with self._lock, self.con:
            self.con.executemany("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)",
                                 [(k, json.dumps(v, ensure_ascii=False)) for k, v in values.items()])

    # ------------------------------------------------------------------ sessions
    def _q(self, sql, args=()):
        with self._lock:
            return [Session(r[0], r[1], r[2], r[3], r[4], bool(r[5]), bool(r[6]), r[7], r[8], bool(r[9]))
                    for r in self.con.execute(sql, args)]

    _COLS = "id, plate, entry_time, exit_time, fee, paid, subscriber, entry_img, exit_img, manual"

    def open_sessions(self):
        return self._q(f"SELECT {self._COLS} FROM sessions WHERE exit_time IS NULL ORDER BY entry_time DESC")

    def open_session(self, plate):
        r = self._q(f"SELECT {self._COLS} FROM sessions WHERE exit_time IS NULL AND plate = ? ORDER BY entry_time DESC LIMIT 1",
                    (plates.normalize(plate),))
        return r[0] if r else None

    def similar_open_sessions(self, plate, max_dist=1):
        """Open sessions whose plate differs by at most `max_dist` characters (one misread character)."""
        p = plates.normalize(plate)
        hits = [(plates.edit_distance(p, s.plate), s) for s in self.open_sessions()]
        return [s for d, s in sorted(hits, key=lambda x: x[0]) if d <= max_dist]

    def session(self, sid):
        r = self._q(f"SELECT {self._COLS} FROM sessions WHERE id = ?", (sid,))
        return r[0] if r else None

    def add_entry(self, plate, t=None, img=None, manual=False):
        with self._lock, self.con:
            p = plates.normalize(plate)
            cur = self.con.execute("INSERT INTO sessions(plate, entry_time, entry_img, subscriber, manual) VALUES (?, ?, ?, ?, ?)",
                                   (p, t or time.time(), img, int(self.subscriber_active(p)), int(manual)))
            return self.session(cur.lastrowid)

    def close_session(self, sid, fee, t=None, img=None, paid=True):
        with self._lock, self.con:
            self.con.execute("UPDATE sessions SET exit_time = ?, fee = ?, paid = ?, exit_img = COALESCE(?, exit_img) WHERE id = ?",
                             (t or time.time(), int(fee), int(paid), img, sid))
        return self.session(sid)

    def update_plate(self, sid, plate):
        with self._lock, self.con:
            self.con.execute("UPDATE sessions SET plate = ? WHERE id = ?", (plates.normalize(plate), sid))

    def delete_session(self, sid):
        with self._lock, self.con:
            self.con.execute("DELETE FROM sessions WHERE id = ?", (sid,))

    def sessions_between(self, t0, t1, search=""):
        """Sessions that entered in [t0, t1), newest first; `search` matches part of the plate."""
        sql = f"SELECT {self._COLS} FROM sessions WHERE entry_time >= ? AND entry_time < ?"
        args = [t0, t1]
        if search:
            sql += " AND plate LIKE ?"
            args.append(f"%{plates.normalize(search)}%")
        return self._q(sql + " ORDER BY entry_time DESC", args)

    def revenue_between(self, t0, t1):
        with self._lock:
            r = self.con.execute("SELECT COALESCE(SUM(fee), 0), COUNT(*) FROM sessions WHERE exit_time >= ? AND exit_time < ? AND paid = 1",
                                 (t0, t1)).fetchone()
            return int(r[0]), int(r[1])

    def entries_between(self, t0, t1):
        with self._lock:
            return int(self.con.execute("SELECT COUNT(*) FROM sessions WHERE entry_time >= ? AND entry_time < ?", (t0, t1)).fetchone()[0])

    # ------------------------------------------------------------------ subscribers
    def subscribers(self, search=""):
        sql, args = "SELECT id, plate, name, phone, valid_until, created FROM subscribers", []
        if search:
            sql += " WHERE plate LIKE ? OR name LIKE ?"
            args = [f"%{plates.normalize(search)}%", f"%{search}%"]
        with self._lock:
            return [Subscriber(*r) for r in self.con.execute(sql + " ORDER BY created DESC", args)]

    def subscriber_active(self, plate):
        with self._lock:
            r = self.con.execute("SELECT valid_until FROM subscribers WHERE plate = ?", (plates.normalize(plate),)).fetchone()
            return bool(r and r[0] >= time.time())

    def save_subscriber(self, plate, name, phone, valid_until, sid=None):
        with self._lock, self.con:
            p = plates.normalize(plate)
            if sid:
                self.con.execute("UPDATE subscribers SET plate = ?, name = ?, phone = ?, valid_until = ? WHERE id = ?",
                                 (p, name, phone, valid_until, sid))
            else:
                self.con.execute("INSERT INTO subscribers(plate, name, phone, valid_until, created) VALUES (?, ?, ?, ?, ?)",
                                 (p, name, phone, valid_until, time.time()))

    def delete_subscriber(self, sid):
        with self._lock, self.con:
            self.con.execute("DELETE FROM subscribers WHERE id = ?", (sid,))

    # ------------------------------------------------------------------ images
    def save_image(self, bgr, tag):
        """Store a plate snapshot (JPEG) and return its path."""
        import cv2

        path = self.images / f"{time.strftime('%Y%m%d')}_{tag}_{time.time_ns()}.jpg"
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])  # imwrite fails on non-ASCII Windows paths
        if not ok:
            return None
        path.write_bytes(buf.tobytes())
        return str(path)
