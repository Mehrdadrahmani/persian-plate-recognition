"""Author tool: create ParkYar activation keys.

    python tools/make_license.py --init              # once: create the private key (~/.parkyar/license_private.key)
    python tools/make_license.py 3F9A-1C22-B07E-54D1 # activation key for the computer ID a customer sends

Keep the private key secret and backed up. It never goes into the repository; without it no new keys can be made.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parkyar import licensing  # noqa: E402

KEY_FILE = Path.home() / ".parkyar" / "license_private.key"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("machine_id", nargs="?", help="computer ID shown in the customer's activation window")
    ap.add_argument("--init", action="store_true", help="create a new private key")
    ap.add_argument("--key-file", type=Path, default=KEY_FILE)
    a = ap.parse_args()
    if a.init:
        if a.key_file.exists():
            sys.exit(f"{a.key_file} already exists; refusing to overwrite it")
        a.key_file.parent.mkdir(parents=True, exist_ok=True)
        seed = os.urandom(32)
        a.key_file.write_text(seed.hex())
        os.chmod(a.key_file, 0o600)
        print(f"private key: {a.key_file}  (back it up; keep it secret)")
        print(f"public key (goes into parkyar/licensing.py): {licensing.public_key(seed).hex()}")
        return
    if not a.machine_id:
        ap.error("give a computer ID, or --init")
    seed = bytes.fromhex(a.key_file.read_text().strip())
    if licensing.public_key(seed) != licensing.PUBLIC_KEY:
        sys.exit("this private key does not match the public key built into the app")
    key = licensing.make_key(seed, a.machine_id)
    assert licensing.key_is_valid(key, a.machine_id.strip().upper())
    print(key)


if __name__ == "__main__":
    main()
