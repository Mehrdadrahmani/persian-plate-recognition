"""Phase 1 data: inspect/clean the LPD detection dataset, EDA, grouped split, Ultralytics layout.

LPD layout: images/*.jpg, labels/*.txt (YOLO: `0 cx cy w h`, normalized), plate_labels.csv (plate text per image,
space-separated when an image has several plates). The raw folder is never modified.
"""
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .plates import normalize_plate, parse_plate, to_latin

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class LPDScan:
    images: dict          # stem -> Path
    plates: dict          # stem -> [normalized plate strings]
    clean_labels: dict    # stem -> cleaned YOLO label text
    bad: dict             # problem -> [stems]
    img_df: pd.DataFrame  # one row per valid image
    box_df: pd.DataFrame  # one row per box (pixel sizes)

    @property
    def valid(self):
        return sorted(self.clean_labels)


def scan_lpd(lpd_dir, log=print):
    """Read every image header + label; skip (and report) broken images, missing/empty/malformed labels."""
    lpd_dir = Path(lpd_dir)
    images = {p.stem: p for p in sorted((lpd_dir / "images").iterdir()) if p.suffix.lower() in IMG_EXTS}
    labels = {p.stem: p for p in sorted((lpd_dir / "labels").glob("*.txt"))}
    plates = {}
    csv = lpd_dir / "plate_labels.csv"
    if csv.is_file():
        for r in pd.read_csv(csv, dtype=str, encoding="utf-8", keep_default_na=False).itertuples():
            plates[Path(r.image_path).stem] = [normalize_plate(t) for t in r.label.split()]
    log(f"LPD: {len(images)} images, {len(labels)} label files, plate text for {len(plates)} images "
        f"(plate text present: {csv.is_file()})")

    bad, clean, img_rows, box_rows = defaultdict(list), {}, [], []
    for stem, ip in images.items():
        try:
            with Image.open(ip) as im:
                w, h = im.size
                im.verify()
        except Exception:
            bad["broken_image"].append(stem)
            continue
        lp = labels.get(stem)
        if lp is None:
            bad["missing_label"].append(stem)
            continue
        lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
        if not lines:
            bad["empty_label"].append(stem)
            continue
        boxes, problem = [], None
        for parts in lines:
            try:
                assert len(parts) == 5 and int(parts[0]) == 0
                cx, cy, bw, bh = map(float, parts[1:])
            except (AssertionError, ValueError):
                problem = "malformed_line"
                break
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            if bw <= 0 or bh <= 0 or min(x1, y1) < -0.01 or max(x2, y2) > 1.01:
                problem = "out_of_range"
                break
            x1, y1, x2, y2 = max(x1, 0.0), max(y1, 0.0), min(x2, 1.0), min(y2, 1.0)  # clip tiny overshoot
            boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        if problem:
            bad[problem].append(stem)
            continue
        clean[stem] = "".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n" for a, b, c, d in boxes)
        img_rows.append({"stem": stem, "w": w, "h": h, "n_boxes": len(boxes)})
        box_rows += [{"stem": stem, "cx": a, "cy": b, "bw": c, "bh": d, "w_px": c * w, "h_px": d * h} for a, b, c, d in boxes]
    box_df = pd.DataFrame(box_rows)
    box_df["aspect"] = box_df.w_px / box_df.h_px
    n_agree = sum(len(plates.get(s, [])) == len(clean[s].splitlines()) for s in clean)
    n_parse = sum(parse_plate(t) is not None for ts in plates.values() for t in ts)
    log(f"valid images: {len(clean)}/{len(images)}, boxes: {len(box_df)}; skipped: {dict((k, len(v)) for k, v in bad.items()) or 'none'}")
    log(f"#plate strings == #boxes for {n_agree}/{len(clean)} images; {n_parse} plate strings parse as 2 digits + letter + 5 digits")
    return LPDScan(images, plates, clean, dict(bad), pd.DataFrame(img_rows), box_df)


def plot_lpd_eda(scan, save, n_samples=6, seed=42):
    """One figure per statistic + a sample grid with ground-truth boxes."""
    img_df, box_df = scan.img_df, scan.box_df
    plt.figure()
    plt.scatter(img_df.w, img_df.h, s=8, alpha=0.3)
    plt.title("LPD image sizes")
    plt.xlabel("width (px)")
    plt.ylabel("height (px)")
    save("lpd_image_sizes")

    bc = Counter(img_df.n_boxes)
    plt.figure()
    plt.bar(list(bc), list(bc.values()))
    for k, v in bc.items():
        plt.text(k, v, str(v), ha="center", va="bottom")
    plt.yscale("log")
    plt.title("number of plates per image")
    plt.xlabel("plates in the image")
    plt.ylabel("images (log scale)")
    save("lpd_plates_per_image")

    for col, title in (("h_px", "height"), ("w_px", "width")):
        plt.figure()
        plt.hist(box_df[col], bins=60)
        plt.axvline(box_df[col].median(), color="r", ls="--", label=f"median {box_df[col].median():.0f} px")
        plt.title(f"plate box {title} (native resolution)")
        plt.xlabel(f"{title} (px)")
        plt.ylabel("plates")
        plt.legend()
        save(f"lpd_box_{title}")

    plt.figure()
    plt.hist(box_df.aspect.clip(0, 12), bins=60)
    plt.title("plate box aspect ratio (width / height)")
    plt.xlabel("w / h")
    plt.ylabel("plates")
    save("lpd_box_aspect_ratio")

    plt.figure(figsize=(9, 8))
    hm = plt.hist2d(box_df.cx, box_df.cy, bins=40, range=[[0, 1], [0, 1]], cmap="viridis")
    plt.gca().invert_yaxis()
    plt.colorbar(hm[3], label="plates")
    plt.title("where plates are in the image (box centres)")
    plt.xlabel("x (normalized)")
    plt.ylabel("y (normalized)")
    plt.grid(False)
    save("lpd_box_centre_heatmap")

    rng = random.Random(seed)
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    for a, stem in zip(axes.ravel(), rng.sample(scan.valid, min(n_samples, len(scan.valid)))):
        img = cv2.cvtColor(cv2.imread(str(scan.images[stem])), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        for line in scan.clean_labels[stem].splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            cv2.rectangle(img, (int((cx - bw / 2) * W), int((cy - bh / 2) * H)), (int((cx + bw / 2) * W), int((cy + bh / 2) * H)),
                          (0, 255, 0), max(3, W // 250))
        a.imshow(img)
        a.set_title(f"{stem}: " + ", ".join(to_latin(t) for t in scan.plates.get(stem, [])), fontsize=13)
        a.axis("off")
    plt.suptitle("LPD samples with ground-truth boxes", fontsize=18)
    save("lpd_samples")


def group_split(stems, plates, fractions, seed=42):
    """70/15/15-style split where images sharing a plate string (same vehicle) stay in one split."""
    parent = {s: s for s in stems}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    first = {}
    for s in stems:
        for t in plates.get(s, []):
            if t in first:
                parent[find(s)] = find(first[t])
            else:
                first[t] = s
    groups = defaultdict(list)
    for s in stems:
        groups[find(s)].append(s)
    group_list = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    random.Random(seed).shuffle(group_list)
    n = len(stems)
    splits = {"train": [], "val": [], "test": []}
    for g in group_list:
        if len(splits["test"]) < fractions["test"] * n:
            splits["test"] += g
        elif len(splits["val"]) < fractions["val"] * n:
            splits["val"] += g
        else:
            splits["train"] += g
    splits = {k: sorted(v) for k, v in splits.items()}
    assert not (set(splits["train"]) & set(splits["test"])) and not (set(splits["val"]) & set(splits["test"]))
    return splits, len(groups)


def write_split_files(splits, images, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in splits.items():
        (out_dir / f"lpd_{k}_split.txt").write_text("".join(f"{images[s].name}\n" for s in v))


def read_split_files(split_dir):
    return {k: [Path(l).stem for l in (Path(split_dir) / f"lpd_{k}_split.txt").read_text().split()]
            for k in ("train", "val", "test") if (Path(split_dir) / f"lpd_{k}_split.txt").is_file()}


def build_yolo_layout(splits, images, clean_labels, yolo_dir, lpd_dir, log=print):
    """images/{split} = symlinks to the raw JPGs, labels/{split} = cleaned label copies. Rebuilt only on change."""
    yolo_dir, lpd_dir = Path(yolo_dir), Path(lpd_dir)
    assert lpd_dir.resolve() not in yolo_dir.resolve().parents, "never write into the raw LPD folder"
    sig = hashlib.md5(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    marker = yolo_dir / ".split_signature"
    if marker.exists() and marker.read_text() == sig:
        log("YOLO layout up to date:", yolo_dir)
        return yolo_dir
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split, stems in splits.items():
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for s in stems:
            src = images[s].resolve()
            dst = yolo_dir / "images" / split / src.name
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
            (yolo_dir / "labels" / split / f"{s}.txt").write_text(clean_labels[s])
    marker.write_text(sig)
    log("built YOLO layout", yolo_dir, {k: len(v) for k, v in splits.items()})
    return yolo_dir


def write_data_yaml(yolo_dir, path):
    Path(path).write_text(f"""# Root directory of the dataset
path: {json.dumps(str(yolo_dir))}

# Relative paths to image folders (relative to 'path' above)
train: images/train
val: images/val
test: images/test

# Number of classes and mapping from class ID to name
nc: 1
names:
  0: license_plate
""")
    return Path(path)
