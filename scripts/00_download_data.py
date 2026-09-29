"""Download and extract the LPD (detection) and LPR (recognition) datasets into data/.

    python scripts/00_download_data.py            # -> data/LPD, data/LPR (skips what is already there)

Source: the Google Drive files linked in the original course notebook.
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from platereader.config import load_config  # noqa: E402

FILES = {"LPD": "1UBV930A9pRhVwop-WunmP38th2XwEumx", "LPR": "1Me3Bk1mmITaf5QQE5Kvk1E1d8xaqsnhf"}


def is_root(p, kind):
    return (p / "images").is_dir() and (p / "labels").is_dir() if kind == "LPD" else \
        (p / "detections").is_dir() and (p / "valid_samples.csv").is_file()


def find_root(base, kind, depth=4):
    level = [base]
    for _ in range(depth + 1):
        nxt = []
        for p in level:
            if is_root(p, kind):
                return p
            nxt += sorted(c for c in p.iterdir() if c.is_dir() and not c.name.startswith(("__MACOSX", ".")))
        level = nxt
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    data_dir = load_config(data_dir=args.data_dir).paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    import gdown

    for kind, fid in FILES.items():
        target = data_dir / kind
        if target.exists() and is_root(target, kind):
            print(f"{kind}: already at {target}")
            continue
        zpath = data_dir / f"{kind}.zip"
        if not zpath.exists():
            gdown.download(id=fid, output=str(zpath), quiet=False)
        tmp = data_dir / f"_{kind}_extract"
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(tmp)
        root = find_root(tmp, kind)
        assert root is not None, f"could not find the {kind} layout inside {zpath}"
        shutil.move(str(root), str(target))
        shutil.rmtree(tmp, ignore_errors=True)
        zpath.unlink()
        print(f"{kind}: ready at {target}")


if __name__ == "__main__":
    main()
