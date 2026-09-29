"""Free trial (a fixed number of plate recognitions) and per-computer activation keys.

An activation key is an Ed25519 signature of this computer's ID, made with the author's private key (kept offline,
never in this repository). The app only holds the public key, so it can check keys but cannot create them.
"""
import base64
import hashlib
import hmac
import os
import subprocess
import sys
import uuid
from pathlib import Path

FREE_RECOGNITIONS = 10
AUTHOR = "Mehrdad Rahmani"
CONTACT = "mehrdad.rahmani100@gmail.com"
PUBLIC_KEY = bytes.fromhex("e88064640531403f925f274fb1bd3e3094a6075ccab490c713934610677420e3")
_CONTEXT = b"ParkYar-license-v1|"

# ------------------------------------------------------------------ Ed25519 (RFC 8032 reference arithmetic)
_p = 2 ** 255 - 19
_q = 2 ** 252 + 27742317777372353535851937790883648493
_d = -121665 * pow(121666, _p - 2, _p) % _p
_sqrt_m1 = pow(2, (_p - 1) // 4, _p)


def _sha512_int(b):
    return int.from_bytes(hashlib.sha512(b).digest(), "little")


def _add(P, Q):
    A = (P[1] - P[0]) * (Q[1] - Q[0]) % _p
    B = (P[1] + P[0]) * (Q[1] + Q[0]) % _p
    C = 2 * P[3] * Q[3] * _d % _p
    D = 2 * P[2] * Q[2] % _p
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _p, G * H % _p, F * G % _p, E * H % _p)


def _mul(s, P):
    Q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            Q = _add(Q, P)
        P = _add(P, P)
        s >>= 1
    return Q


def _equal(P, Q):
    return (P[0] * Q[2] - Q[0] * P[2]) % _p == 0 and (P[1] * Q[2] - Q[1] * P[2]) % _p == 0


def _recover_x(y, sign):
    if y >= _p:
        return None
    x2 = (y * y - 1) * pow(_d * y * y + 1, _p - 2, _p)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_p + 3) // 8, _p)
    if (x * x - x2) % _p:
        x = x * _sqrt_m1 % _p
    if (x * x - x2) % _p:
        return None
    return _p - x if (x & 1) != sign else x


_gy = 4 * pow(5, _p - 2, _p) % _p
_gx = _recover_x(_gy, 0)
_G = (_gx, _gy, 1, _gx * _gy % _p)


def _compress(P):
    zi = pow(P[2], _p - 2, _p)
    x, y = P[0] * zi % _p, P[1] * zi % _p
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(b):
    y = int.from_bytes(b, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _p)


def _expand(seed):
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a = (a & ((1 << 254) - 8)) | (1 << 254)
    return a, h[32:]


def public_key(seed):
    return _compress(_mul(_expand(seed)[0], _G))


def sign(seed, msg):
    a, prefix = _expand(seed)
    A = _compress(_mul(a, _G))
    r = _sha512_int(prefix + msg) % _q
    R = _compress(_mul(r, _G))
    s = (r + _sha512_int(R + A + msg) % _q * a) % _q
    return R + int.to_bytes(s, 32, "little")


def verify(public, msg, sig):
    if len(public) != 32 or len(sig) != 64:
        return False
    A, R = _decompress(public), _decompress(sig[:32])
    if A is None or R is None:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= _q:
        return False
    h = _sha512_int(sig[:32] + public + msg) % _q
    return _equal(_mul(s, _G), _add(R, _mul(h, A)))


# ------------------------------------------------------------------ computer ID and key format
def _raw_machine_id():
    try:
        if sys.platform == "win32":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0,
                               winreg.KEY_READ | winreg.KEY_WOW64_64KEY)
            return winreg.QueryValueEx(k, "MachineGuid")[0]
        if sys.platform == "darwin":
            out = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True, text=True).stdout
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split('"')[-2]
        for f in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            if os.path.exists(f):
                return Path(f).read_text().strip()
    except Exception:
        pass
    return str(uuid.getnode())


def machine_id():
    """Short, stable ID of this computer, e.g. 3F9A-1C22-B07E-54D1."""
    h = hashlib.sha256(b"parkyar:" + _raw_machine_id().encode()).hexdigest()[:16].upper()
    return "-".join(h[i:i + 4] for i in range(0, 16, 4))


def format_key(sig):
    s = base64.b32encode(sig).decode().rstrip("=")
    return "-".join(s[i:i + 5] for i in range(0, len(s), 5))


def parse_key(text):
    s = "".join(ch for ch in text.upper() if ch.isalnum())
    try:
        return base64.b32decode(s + "=" * (-len(s) % 8))
    except Exception:
        return b""


def key_is_valid(key_text, mid=None, public=None):
    mid = mid or machine_id()
    return verify(public or PUBLIC_KEY, _CONTEXT + mid.encode(), parse_key(key_text))


def make_key(seed, mid):
    """Author side: activation key for a computer ID (needs the private seed)."""
    return format_key(sign(seed, _CONTEXT + mid.strip().upper().encode()))


# ------------------------------------------------------------------ trial counter and activation state
class License:
    """Counts recognitions in several places (data folder, home folder, Windows registry) and uses the highest
    value, so deleting one of them does not reset the trial. An activation key lifts the limit on this computer."""

    def __init__(self, data_dir, use_registry=True, home=None):
        self.data_dir = Path(data_dir)
        self.home = Path(home) if home else Path.home()
        self.use_registry = use_registry and sys.platform == "win32"
        self.mid = machine_id()
        self._key_files = [self.data_dir / "license.key", self.home / ".parkyar_license"]
        self.activated = self._load_key()
        self.used = self._load_count()

    # --- activation
    def _load_key(self):
        for f in self._key_files:
            try:
                if key_is_valid(f.read_text(encoding="utf-8"), self.mid):
                    return True
            except OSError:
                pass
        return False

    def activate(self, key_text):
        if not key_is_valid(key_text, self.mid):
            return False
        for f in self._key_files:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(key_text.strip(), encoding="utf-8")
            except OSError:
                pass
        self.activated = True
        return True

    # --- trial counter
    def _tag(self, n):
        return hmac.new(self.mid.encode(), f"parkyar-usage:{n}".encode(), hashlib.sha256).hexdigest()[:16]

    def _files(self):
        return [self.data_dir / ".usage", self.home / ".parkyar_usage"]

    def _decode(self, s):
        try:
            n, tag = s.strip().split(":")
            return int(n) if hmac.compare_digest(tag, self._tag(int(n))) else FREE_RECOGNITIONS
        except Exception:
            return FREE_RECOGNITIONS  # edited or corrupt: treat the trial as used up

    def _load_count(self):
        values = []
        for f in self._files():
            if f.exists():
                values.append(self._decode(f.read_text(encoding="utf-8", errors="ignore")))
        if self.use_registry:
            try:
                import winreg

                k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\ParkYar")
                values.append(self._decode(winreg.QueryValueEx(k, "u")[0]))
            except OSError:
                pass
        return max(values, default=0)

    def _save_count(self):
        s = f"{self.used}:{self._tag(self.used)}"
        for f in self._files():
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(s, encoding="utf-8")
            except OSError:
                pass
        if self.use_registry:
            try:
                import winreg

                k = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\ParkYar")
                winreg.SetValueEx(k, "u", 0, winreg.REG_SZ, s)
            except OSError:
                pass

    @property
    def remaining(self):
        return max(0, FREE_RECOGNITIONS - self.used)

    def can_recognize(self):
        return self.activated or self.remaining > 0

    def consume(self):
        if not self.activated:
            self.used += 1
            self._save_count()
