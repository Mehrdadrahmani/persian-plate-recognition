"""Phase 1/2 evaluation helpers: recognizer test metrics, comparison, latency, figures."""
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import recognizer as R
from .plates import DIGIT_POS, LETTER_LATIN, LETTER_POS, PLATE_LEN, to_latin
from .train_recognizer import count_params, evaluate
from .utils import sync_device


def load_recognizer(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = R.build_model(ck["arch"], ck["n_letters"], pretrained=False, backbone=ck.get("backbone") or "mobilenet_v3_large")
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def recognizer_metrics(model, loader, device, encoder):
    """plate/char/letter accuracy, macro F1 (digits, letters), loss, per-position accuracy + raw predictions."""
    from sklearn.metrics import f1_score

    r = evaluate(model, loader, device, return_preds=True)
    P, Y = r.pop("preds"), r.pop("targets")
    r["letter_acc"] = float((P[:, LETTER_POS] == Y[:, LETTER_POS]).float().mean())
    r["digit_macro_f1"] = float(f1_score(Y[:, DIGIT_POS].ravel(), P[:, DIGIT_POS].ravel(), labels=list(range(10)),
                                         average="macro", zero_division=0))
    r["letter_macro_f1"] = float(f1_score(Y[:, LETTER_POS], P[:, LETTER_POS], labels=sorted(set(Y[:, LETTER_POS].tolist())),
                                          average="macro", zero_division=0))
    return r, P, Y


def time_inference(model, dev, batch, warmup=10, iters=50):
    model = model.to(dev).eval()
    x = torch.randn(batch, 3, *R.INPUT_SIZE, device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync_device(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync_device(dev)
    return (time.perf_counter() - t0) / iters * 1000


def comparison_table(models, results, ckpt_paths, device, timing):
    rows = []
    for name, model in models.items():
        total, trainable = count_params(model)
        r = results[name]
        row = {"model": name, "test_plate_acc": r["plate_acc"], "test_char_acc": r["char_acc"], "test_letter_acc": r["letter_acc"],
               "digit_macro_f1": r["digit_macro_f1"], "letter_macro_f1": r["letter_macro_f1"], "test_loss": r["loss"],
               "params_total": total, "params_trainable": trainable, "size_MB": ckpt_paths[name].stat().st_size / 2 ** 20}
        cpu = copy.deepcopy(model).cpu()
        if device.type != "cpu":
            row[f"{device.type}_ms_per_image"] = time_inference(model, device, 1, timing["warmup"], timing["iters"])
            row[f"{device.type}_ms_per_batch{timing['batch']}"] = time_inference(model, device, timing["batch"], timing["warmup"], timing["iters"])
        row["cpu_ms_per_image"] = time_inference(cpu, torch.device("cpu"), 1, timing["warmup"], timing["iters"])
        row[f"cpu_ms_per_batch{timing['batch']}"] = time_inference(cpu, torch.device("cpu"), timing["batch"], timing["warmup"], timing["iters"])
        model.to(device)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_letter_confusion(P, Y, letter_vocab, name, save):
    from sklearn.metrics import confusion_matrix

    cls = sorted(set(Y[:, LETTER_POS].tolist()) | set(P[:, LETTER_POS].tolist()))
    cm = confusion_matrix(Y[:, LETTER_POS], P[:, LETTER_POS], labels=cls)
    names = [LETTER_LATIN[letter_vocab[i]] for i in cls]
    plt.figure(figsize=(13, 11))
    plt.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="fraction of the true class")
    for (ti, pi), v in np.ndenumerate(cm):
        if v:
            plt.text(pi, ti, str(v), ha="center", va="center", fontsize=9, color="white" if v / max(cm[ti].sum(), 1) > 0.5 else "black")
    plt.xticks(range(len(cls)), names, rotation=70)
    plt.yticks(range(len(cls)), [f"{n} ({cm[k].sum()})" for k, n in enumerate(names)])
    plt.grid(False)
    plt.xlabel("predicted letter")
    plt.ylabel("true letter (test count)")
    plt.title(f"{name}: letter confusion on the test set")
    save(f"lpr_{name}_letter_confusion")


def plot_misclassified(P, Y, files, img_dir, encoder, name, save, n=12):
    wrong = torch.nonzero(~(P == Y).all(1)).ravel().tolist()
    if not wrong:
        return
    fig, axes = plt.subplots(6, 2, figsize=(16, 16))
    for a in axes.ravel():
        a.axis("off")
    for a, i in zip(axes.ravel(), wrong[:n]):
        a.imshow(Image.open(img_dir / files[i]))
        a.set_title(f"true {to_latin(encoder.decode(Y[i]))}   pred {to_latin(encoder.decode(P[i]))}", fontsize=13)
    plt.suptitle(f"{name}: {len(wrong)} misclassified test plates ({len(wrong) / len(Y):.1%}), first {min(n, len(wrong))}", fontsize=18)
    save(f"lpr_{name}_misclassified")


def plot_split_bars(table, cols, labels, title, name, save):
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    x = np.arange(len(cols))
    plt.figure(figsize=(13, 6.5))
    for k, split in enumerate(table.index):
        v = table.loc[split, cols].to_numpy(float)
        plt.bar(x + (k - 1) * 0.27, v, 0.27, color=colors.get(split), label=split)
        for xi, vi in zip(x + (k - 1) * 0.27, v):
            plt.text(xi, vi, f"{vi:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, labels)
    plt.ylim(0, 1.1)
    plt.ylabel("score")
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    save(name)


def verdict(train, val, test):
    if max(train, val, test) < 0.5:
        return "UNDERFITTING (all splits low)"
    gap = train - test
    return f"possible OVERFITTING (train {gap:.1%} above test)" if gap > 0.05 else f"generalizes well (train-test gap {gap:+.1%})"


__all__ = ["load_recognizer", "recognizer_metrics", "comparison_table", "plot_letter_confusion", "plot_misclassified",
           "plot_split_bars", "verdict", "PLATE_LEN"]
