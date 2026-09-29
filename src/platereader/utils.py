"""Shared helpers: device selection, seeding, logging, figure saving, GPU memory."""
import gc
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported MPS ops fall back to CPU

import matplotlib  # noqa: E402

if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None and not os.environ.get("JPY_PARENT_PID"):
    matplotlib.use("Agg")  # headless scripts: save figures, don't open windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

plt.rcParams.update({"figure.figsize": (11, 6.5), "figure.dpi": 100, "savefig.dpi": 150, "font.size": 13,
                     "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12, "axes.grid": True, "grid.alpha": 0.3})


def pick_device(name=None):
    """torch.device: explicit name, else cuda > mps > cpu."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_device(device, all_gpus=False):
    """Ultralytics device argument: 0 / [0, 1, ...] on CUDA, 'mps', or 'cpu'."""
    if device.type == "cuda":
        n = torch.cuda.device_count()
        return list(range(n)) if (all_gpus and n > 1) else 0
    return device.type


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def sync_device(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class RunLog:
    """print() + append timestamped lines to a text log file."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


class FigureSaver:
    """save(name): write the current figure to <dir>/<name>.png (150 dpi), show it in notebooks, close it."""

    def __init__(self, fig_dir, show=None):
        self.dir = Path(fig_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.show = matplotlib.get_backend().lower() not in ("agg",) if show is None else show

    def __call__(self, name):
        plt.tight_layout()
        plt.savefig(self.dir / f"{name}.png", bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close("all")
