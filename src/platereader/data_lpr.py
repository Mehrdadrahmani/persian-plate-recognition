"""Phase 2 data: the LPR recognition dataset (cropped plates + labels), EDA, leakage-safe split, PyTorch dataset.

LPR layout: detections/*.jpg + valid_samples.csv (image_path,label). Labels are normalized: the ZWJ after 'ه' is
removed and the spelled-out 'الف' is one letter token, so every label is 2 digits + letter + 5 digits.
"""
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

from . import recognizer as R
from .plates import DIGIT_VOCAB, DIGIT_POS, LETTER_LATIN, LETTER_POS, normalize_plate, parse_plate, to_latin

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_lpr(lpr_dir, log=print, read_sizes=True):
    """valid_samples.csv -> DataFrame(filename, raw_label, text, digits, letter[, w, h, aspect]) + letter vocabulary."""
    lpr_dir = Path(lpr_dir)
    raw = pd.read_csv(lpr_dir / "valid_samples.csv", dtype=str, encoding="utf-8", keep_default_na=False)
    files = {p.name for p in (lpr_dir / "detections").iterdir() if p.suffix.lower() in IMG_EXTS}
    df = raw.rename(columns={"image_path": "filename", "label": "raw_label"})
    df["text"] = df.raw_label.map(normalize_plate)
    parsed = df.text.map(parse_plate)
    drop = {"missing_file": ~df.filename.isin(files), "duplicate_row": df.filename.duplicated(), "bad_format": parsed.isna()}
    keep = ~(drop["missing_file"] | drop["duplicate_row"] | drop["bad_format"])
    log(f"LPR: {len(raw)} rows, {len(files)} files; dropped " + ", ".join(f"{k}={int(m.sum())}" for k, m in drop.items())
        + f"; normalized {int((df.text != df.raw_label).sum())} labels (ZWJ removed)")
    df = df[keep].reset_index(drop=True)
    df["digits"] = [p[0] for p in parsed[keep]]
    df["letter"] = [p[1] for p in parsed[keep]]
    letter_vocab = [c for c in LETTER_LATIN if c in set(df.letter)]
    if read_sizes:
        sizes = []
        for f in df.filename:
            with Image.open(lpr_dir / "detections" / f) as im:
                sizes.append(im.size)
        df["w"], df["h"] = zip(*sizes)
        df["aspect"] = df.w / df.h
    return df, letter_vocab


def plot_lpr_eda(df, letter_vocab, lpr_dir, save, seed=42):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(4, 3, figsize=(18, 10))
    for a, i in zip(axes.ravel(), rng.choice(len(df), min(12, len(df)), replace=False)):
        r = df.iloc[i]
        a.imshow(Image.open(Path(lpr_dir) / "detections" / r.filename))
        a.set_title(f"{to_latin(r.text)}  ({r.w}x{r.h})", fontsize=14)
        a.axis("off")
    plt.suptitle("random LPR samples (label transliterated: DD-letter-DDD-region)", fontsize=18)
    save("lpr_samples")

    plt.figure()
    for col, lab in (("w", "width"), ("h", "height")):
        plt.hist(df[col], bins=np.logspace(np.log10(df[col].min()), np.log10(df[col].max()), 60), alpha=0.7, label=lab)
    plt.xscale("log")
    plt.title("LPR crop sizes")
    plt.xlabel("pixels (log scale)")
    plt.ylabel("crops")
    plt.legend()
    save("lpr_crop_sizes")

    plt.figure()
    plt.hist(df.aspect, bins=80)
    plt.axvline(df.aspect.median(), color="k", ls=":", label=f"median {df.aspect.median():.2f}")
    plt.axvline(4.0, color="r", ls="--", label="model input 256/64 = 4")
    plt.title("LPR crop aspect ratio (width / height)")
    plt.xlabel("w / h")
    plt.ylabel("crops")
    plt.legend()
    save("lpr_crop_aspect_ratio")

    lc = df.letter.value_counts()
    plt.figure(figsize=(14, 6.5))
    plt.bar([LETTER_LATIN[c] for c in letter_vocab], [lc[c] for c in letter_vocab])
    for i, c in enumerate(letter_vocab):
        plt.text(i, lc[c], str(lc[c]), ha="center", va="bottom", fontsize=10)
    plt.yscale("log")
    plt.xticks(rotation=60)
    plt.title("letter class frequency (all LPR samples)")
    plt.ylabel("samples (log scale)")
    save("lpr_letter_frequency")

    pos = np.zeros((7, 10), int)
    for d in df.digits:
        for j, ch in enumerate(d):
            pos[j, int(ch)] += 1
    plt.figure(figsize=(12, 6.5))
    plt.imshow(pos, cmap="Blues", aspect="auto")
    plt.colorbar(label="count")
    plt.grid(False)
    plt.yticks(range(7), [f"slot {p}" for p in DIGIT_POS])
    plt.xticks(range(10), DIGIT_VOCAB)
    plt.xlabel("digit")
    plt.title("digit frequency per label slot (slot 2 is the letter)")
    save("lpr_digit_position_freq")


def _strat_key(s, min_count):
    c = s.map(s.value_counts())
    k = s.where(c >= min_count, "rare")
    return k if k.value_counts().min() >= 2 else None


def split_lpr(df, lpd_test_plates, holdout=0.2, seed=42, log=print):
    """Split by plate string (a car never straddles splits), stratified on the letter.
    Every plate that appears in an LPD *test* image goes to the LPR test split (Phase 3 stays leakage-free)."""
    from sklearn.model_selection import train_test_split

    g = df.groupby("text")["letter"].first()
    forced = g.index.isin(lpd_test_plates)
    rest = g[~forced]
    g_train, g_hold = train_test_split(rest.index, test_size=holdout, random_state=seed, stratify=_strat_key(rest, 10))
    hold = rest[g_hold]
    g_val, g_test = train_test_split(hold.index, test_size=0.5, random_state=seed, stratify=_strat_key(hold, 2))
    split_of = {**{x: "train" for x in g_train}, **{x: "val" for x in g_val}, **{x: "test" for x in g_test},
                **{x: "test" for x in g.index[forced]}}
    df = df.copy()
    df["split"] = df.text.map(split_of)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(df.text[df.split == a]) & set(df.text[df.split == b])
    log(f"LPR split: {df.split.value_counts().to_dict()}; LPD-test plates forced into test: {int(forced.sum())} plates, "
        f"{int(df.text.isin(lpd_test_plates).sum())} crops")
    return df


class Encoder:
    """Label <-> target LongTensor[8]: slot 2 indexes the letter vocabulary, the other slots index 0-9."""

    def __init__(self, letter_vocab):
        self.letter_vocab = list(letter_vocab)
        self.letter_to_idx = {c: i for i, c in enumerate(self.letter_vocab)}

    def encode(self, text):
        digits, letter = parse_plate(text)
        idx = [DIGIT_VOCAB.index(c) for c in digits]
        return idx[:2] + [self.letter_to_idx[letter]] + idx[2:]

    def decode(self, target):
        t = [int(v) for v in target]
        return "".join(DIGIT_VOCAB[i] for i in t[:2]) + self.letter_vocab[t[2]] + "".join(DIGIT_VOCAB[i] for i in t[3:])


def random_crop_pad(img, max_frac):
    """Move each side in/out by up to max_frac of the size (edge-replicate when growing): imitates YOLO box errors."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    l, t, r, b = (int(round(random.uniform(-max_frac, max_frac) * s)) for s in (w, h, w, h))
    a = np.pad(a, ((max(t, 0), max(b, 0)), (max(l, 0), max(r, 0)), (0, 0)), mode="edge")
    H2, W2 = a.shape[:2]
    a = a[max(-t, 0): H2 - max(-b, 0), max(-l, 0): W2 - max(-r, 0)]
    return Image.fromarray(a) if a.shape[0] >= 4 and a.shape[1] >= 4 else img


class TrainAugment:
    """Train-only: crop jitter -> letterbox -> small affine + photometric noise (no horizontal flip)."""

    def __init__(self, jitter):
        self.jitter = jitter
        self.tf = T.Compose([
            T.RandomApply([T.RandomAffine(degrees=4, translate=(0.03, 0.05), scale=(0.9, 1.05),
                                          shear=(-6, 6, -2, 2), fill=R.PAD_VALUE)], p=0.7),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
            T.RandomGrayscale(p=0.1),
        ])

    def __call__(self, img):
        return self.tf(R.letterbox(random_crop_pad(img, self.jitter)))


class PLPRDataset(data.Dataset):
    """Cropped plates -> (image tensor [3, 64, 256], target LongTensor[8])."""

    def __init__(self, df=None, img_dir=None, encoder=None, augment=None):
        self.files = [] if df is None else df.filename.tolist()
        self.labels = [] if df is None else df.text.tolist()
        self.img_dir = Path(img_dir) if img_dir is not None else None
        self.augment = augment
        self.targets = [torch.tensor(encoder.encode(t), dtype=torch.long) for t in self.labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(self.img_dir / self.files[idx]) as im:
            img = im.convert("RGB")
        img = self.augment(img) if self.augment is not None else R.letterbox(img)
        return R.to_tensor(img), self.targets[idx]


def make_loader(ds, batch_size, train, num_workers=0, pin_memory=False, seed=42):
    return data.DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers,
                           pin_memory=pin_memory, persistent_workers=num_workers > 0,
                           generator=torch.Generator().manual_seed(seed) if train else None)


def letter_counts(df, letter_vocab):
    c = Counter(df.letter)
    return [c.get(x, 0) for x in letter_vocab]


__all__ = ["load_lpr", "plot_lpr_eda", "split_lpr", "Encoder", "TrainAugment", "PLPRDataset", "make_loader",
           "random_crop_pad", "LETTER_POS", "DIGIT_POS"]
