"""Phase 2: shared training loop for both recognizers (AdamW + OneCycle, AMP on CUDA, early stopping, resume,
gradient-health logging) plus history plots and reports."""
import contextlib
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from .plates import DIGIT_POS, DIGIT_VOCAB, LETTER_POS, PLATE_LEN


def letter_class_weights(train_df, letter_vocab, power=0.5, max_w=5.0):
    counts = train_df.letter.value_counts().reindex(letter_vocab, fill_value=0).to_numpy(float)
    w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** power, 0.0)
    w = np.clip(w / w[counts > 0].mean(), 0.0, max_w)
    return torch.tensor(w, dtype=torch.float32)


class PlateLoss(nn.Module):
    """Mean of the 8 per-slot cross-entropies (7 digit slots + 1 letter slot)."""

    def __init__(self, letter_weights=None, label_smoothing=0.0):
        super().__init__()
        self.digit_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.letter_ce = nn.CrossEntropyLoss(weight=letter_weights, label_smoothing=label_smoothing)

    def forward(self, outputs, target):
        digit_logits, letter_logits = outputs
        d = self.digit_ce(digit_logits.float().reshape(-1, len(DIGIT_VOCAB)), target[:, DIGIT_POS].reshape(-1))
        l = self.letter_ce(letter_logits.float(), target[:, LETTER_POS])
        return (len(DIGIT_POS) * d + l) / PLATE_LEN


def predict_targets(outputs):
    digit_logits, letter_logits = outputs
    pred = torch.empty(digit_logits.shape[0], PLATE_LEN, dtype=torch.long, device=digit_logits.device)
    pred[:, DIGIT_POS] = digit_logits.argmax(-1)
    pred[:, LETTER_POS] = letter_logits.argmax(-1)
    return pred


def autocast_ctx(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, return_preds=False):
    criterion = criterion or PlateLoss().to(device)
    model.eval()
    tot, n, preds, targets = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(device):
            out = model(x)
        out = tuple(o.float() for o in out)
        tot += criterion(out, y).item() * len(y)
        n += len(y)
        preds.append(predict_targets(out).cpu())
        targets.append(y.cpu())
    P, Y = torch.cat(preds), torch.cat(targets)
    ok = P == Y
    res = {"loss": tot / n, "char_acc": ok.float().mean().item(), "plate_acc": ok.all(1).float().mean().item(),
           "pos_acc": ok.float().mean(0).tolist()}
    if return_preds:
        res["preds"], res["targets"] = P, Y
    return res


def param_groups(module, lr, weight_decay):
    decay, no_decay = [], []
    for _, p in module.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "lr": lr, "weight_decay": weight_decay}, {"params": no_decay, "lr": lr, "weight_decay": 0.0}]


def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


def _monitor_layers(model):
    convs = [n for n, p in model.named_parameters() if p.ndim == 4 and not n.startswith("head.")]
    return [convs[0], convs[len(convs) // 2], convs[-1], "head.fc.weight", "head.letter_head.weight"]


def _activation_modules(model):
    relus = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.ReLU) and not n.startswith("head.")]
    picks = dict([relus[0], relus[len(relus) // 2], relus[-1]]) if relus else {}
    picks.update({"head.reduce.2": model.head.reduce[2], "head.act": model.head.act})
    return picks


class ActivationMonitor:
    """Forward hooks: fraction of zero (dead) ReLU outputs, recorded on logging steps only."""

    def __init__(self, modules):
        self.active, self.stats = False, {}
        self.handles = [m.register_forward_hook(self._hook(n)) for n, m in modules.items()]

    def _hook(self, name):
        def fn(_m, _i, out):
            if self.active:
                self.stats[name] = (out.detach() <= 0).float().mean().item()
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def _new_history(layer_names, act_names):
    keys = ["epoch", "train_loss", "val_loss", "train_char_acc", "val_char_acc", "train_plate_acc", "val_plate_acc",
            "val_pos_acc", "lr", "epoch_time"]
    return {**{k: [] for k in keys}, "steps": {"step": [], "grad_norm": [], "lr": [], "saturated": [],
            "layers": {n: [] for n in layer_names}, "act_zero": {n: [] for n in act_names}}}


def fit(model, run_dir, loaders, groups, epochs, patience, device, criterion, grad_clip=5.0, pct_start=0.15,
        log_every=10, init_state=None, log=print):
    """Train with AdamW + OneCycle; checkpoint every epoch (resume), keep best val full-plate accuracy.
    Returns (history, best_checkpoint_path); the model ends with the best weights."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    name = run_dir.name
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    model.to(device)
    optimizer = optim.AdamW(groups)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups], epochs=epochs,
                                              steps_per_epoch=len(loaders["train"]), pct_start=pct_start)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    layer_names, act_modules = _monitor_layers(model), _activation_modules(model)
    history = _new_history(layer_names, list(act_modules))
    start_epoch, best_acc, best_epoch, bad_epochs, global_step = 0, -1.0, -1, 0, 0

    if last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        history, best_acc, best_epoch, bad_epochs = ck["history"], ck["best_acc"], ck["best_epoch"], ck["bad_epochs"]
        if ck.get("done"):
            model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True)["state_dict"])
            log(f"[{name}] already finished: best val plate acc {best_acc:.4f} (epoch {best_epoch + 1}) -> loaded")
            return history, best_path
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, global_step = ck["epoch"] + 1, ck["global_step"]
        log(f"[{name}] resuming at epoch {start_epoch + 1}/{epochs}")
    elif init_state is not None:
        model.load_state_dict(init_state)

    eval_criterion = PlateLoss().to(device)
    params = dict(model.named_parameters())
    trainable = [p for g in groups for p in g["params"]]
    mon = ActivationMonitor(act_modules)
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            t0 = time.time()
            loss_sum = torch.zeros((), device=device)
            char_ok = torch.zeros((), device=device)
            plate_ok = torch.zeros((), device=device)
            seen = 0
            for x, y in tqdm(loaders["train"], desc=f"{name} {epoch + 1}/{epochs}", leave=False, mininterval=30):
                log_now = global_step % log_every == 0
                mon.active = log_now
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(device):
                    out = model(x)
                    loss = criterion(out, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if log_now:
                    lnorms = {n: params[n].grad.norm().item() if params[n].grad is not None else float("nan") for n in layer_names}
                gnorm = nn.utils.clip_grad_norm_(trainable, grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    ok = predict_targets(tuple(o.detach() for o in out)) == y
                    char_ok += ok.float().sum()
                    plate_ok += ok.all(1).sum()
                    loss_sum += loss.detach() * len(y)
                    seen += len(y)
                if log_now:
                    st = history["steps"]
                    st["step"].append(global_step)
                    st["grad_norm"].append(gnorm.item())
                    st["lr"].append([g["lr"] for g in optimizer.param_groups])
                    st["saturated"].append((torch.softmax(out[1].detach().float(), -1).amax(-1) > 0.99).float().mean().item())
                    for n in layer_names:
                        st["layers"][n].append(lnorms[n])
                    for n in st["act_zero"]:
                        st["act_zero"][n].append(mon.stats.get(n, float("nan")))
                global_step += 1
            mon.active = False
            val = evaluate(model, loaders["val"], device, eval_criterion)
            h = history
            h["epoch"].append(epoch + 1)
            h["train_loss"].append(loss_sum.item() / seen)
            h["train_char_acc"].append(char_ok.item() / (seen * PLATE_LEN))
            h["train_plate_acc"].append(plate_ok.item() / seen)
            for k in ("loss", "char_acc", "plate_acc", "pos_acc"):
                h[f"val_{k}"].append(val[k])
            h["lr"].append([g["lr"] for g in optimizer.param_groups])
            h["epoch_time"].append(time.time() - t0)
            if val["plate_acc"] > best_acc:
                best_acc, best_epoch, bad_epochs = val["plate_acc"], epoch, 0
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "epoch": epoch, "val": {k: val[k] for k in ("loss", "char_acc", "plate_acc")}}, best_path)
            else:
                bad_epochs += 1
            done = bad_epochs >= patience or epoch == epochs - 1
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None, "epoch": epoch,
                        "global_step": global_step, "history": history, "best_acc": best_acc, "best_epoch": best_epoch,
                        "bad_epochs": bad_epochs, "done": done}, last_path)
            log(f"[{name}] ep {epoch + 1:3d}/{epochs} | loss {h['train_loss'][-1]:.4f}/{val['loss']:.4f} | "
                f"char {h['train_char_acc'][-1]:.4f}/{val['char_acc']:.4f} | plate {h['train_plate_acc'][-1]:.4f}/"
                f"{val['plate_acc']:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | {h['epoch_time'][-1]:.0f}s"
                + ("  *best*" if bad_epochs == 0 else ""))
            if done:
                if bad_epochs >= patience:
                    log(f"[{name}] early stop: no val plate-acc improvement for {patience} epochs")
                break
    finally:
        mon.remove()
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True)["state_dict"])
    log(f"[{name}] best val plate acc {best_acc:.4f} at epoch {best_epoch + 1}")
    return history, best_path


def concat_histories(h1, h2):
    """Join stage-1 and stage-2 histories (epochs / steps continue) for plotting."""
    out = copy.deepcopy(h1)
    e0 = h1["epoch"][-1] if h1["epoch"] else 0
    s0 = h1["steps"]["step"][-1] + 1 if h1["steps"]["step"] else 0
    for k, v in h2.items():
        if k != "steps":
            out[k] += [e + e0 for e in v] if k == "epoch" else v
    for k in ("grad_norm", "saturated"):
        out["steps"][k] += h2["steps"][k]
    out["steps"]["step"] += [s + s0 for s in h2["steps"]["step"]]
    out["steps"]["lr"] = [l[:1] + l[-1:] for l in out["steps"]["lr"] + h2["steps"]["lr"]]
    for grp in ("layers", "act_zero"):
        for n, v in h2["steps"][grp].items():
            out["steps"][grp].setdefault(n, [float("nan")] * (len(out["steps"]["step"]) - len(v)))
            out["steps"][grp][n] += v
        for n in out["steps"][grp]:
            if n not in h2["steps"][grp]:
                out["steps"][grp][n] += [float("nan")] * len(h2["steps"]["step"])
    return out


def save_history_csv(h, out_dir, name):
    out_dir = Path(out_dir)
    ep = pd.DataFrame({k: h[k] for k in ("epoch", "train_loss", "val_loss", "train_char_acc", "val_char_acc",
                                         "train_plate_acc", "val_plate_acc", "epoch_time")})
    ep["lr_group0"] = [l[0] for l in h["lr"]]
    for i in range(PLATE_LEN):
        ep[f"val_acc_slot{i}"] = [pa[i] for pa in h["val_pos_acc"]]
    ep.to_csv(out_dir / f"lpr_{name}_history.csv", index=False)
    st = h["steps"]
    pd.DataFrame({"step": st["step"], "grad_norm": st["grad_norm"], "lr_group0": [l[0] for l in st["lr"]],
                  "letter_softmax_gt_0.99": st["saturated"], **{f"grad_{n}": v for n, v in st["layers"].items()},
                  **{f"zero_frac_{n}": v for n, v in st["act_zero"].items()}}).to_csv(out_dir / f"lpr_{name}_steps.csv", index=False)


def plot_history(h, title, name, save, grad_clip=5.0):
    """Six separate figures: loss, accuracy, LR schedule, global and per-layer gradient norms, activations."""
    st = h["steps"]
    plt.figure()
    plt.plot(h["epoch"], h["train_loss"], "o-", label="train (weighted, smoothed CE)")
    plt.plot(h["epoch"], h["val_loss"], "s-", label="val (plain CE)")
    plt.title(f"{title}: loss per epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    save(f"lpr_{name}_loss")

    plt.figure()
    for k, s in (("char", "-"), ("plate", "--")):
        lab = "per-char" if k == "char" else "full-plate"
        plt.plot(h["epoch"], h[f"train_{k}_acc"], "o" + s, label=f"train {lab}")
        plt.plot(h["epoch"], h[f"val_{k}_acc"], "s" + s, label=f"val {lab}")
    plt.title(f"{title}: accuracy per epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.02)
    plt.legend()
    save(f"lpr_{name}_accuracy")

    plt.figure()
    lrs = np.array(st["lr"], dtype=float)
    for j in range(lrs.shape[1] if lrs.ndim == 2 else 0):
        plt.plot(st["step"], lrs[:, j], label=f"param group {j}")
    plt.title(f"{title}: learning-rate schedule")
    plt.xlabel("training step")
    plt.ylabel("learning rate")
    plt.legend()
    save(f"lpr_{name}_lr_schedule")

    plt.figure()
    plt.plot(st["step"], st["grad_norm"], lw=1)
    plt.axhline(grad_clip, color="r", ls="--", label="clip threshold")
    plt.yscale("log")
    plt.title(f"{title}: global gradient norm (before clipping)")
    plt.xlabel("training step")
    plt.ylabel("L2 norm (log scale)")
    plt.legend()
    save(f"lpr_{name}_grad_norm")

    plt.figure()
    for n, v in st["layers"].items():
        plt.plot(st["step"], v, lw=1, label=n)
    plt.yscale("log")
    plt.title(f"{title}: per-layer gradient norm")
    plt.xlabel("training step")
    plt.ylabel("L2 norm (log scale)")
    plt.legend(fontsize=10)
    save(f"lpr_{name}_layer_grad_norms")

    plt.figure()
    for n, v in st["act_zero"].items():
        plt.plot(st["step"], v, lw=1, label=f"zero fraction {n}")
    plt.plot(st["step"], st["saturated"], "k--", lw=1.2, label="letter softmax > 0.99")
    plt.title(f"{title}: dead activations and output saturation")
    plt.xlabel("training step")
    plt.ylabel("fraction")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=10)
    save(f"lpr_{name}_activations")


def grad_health_report(h, name, grad_clip=5.0, log=print):
    st = h["steps"]
    g = np.array(st["grad_norm"], dtype=float)
    fin = np.isfinite(g)
    log(f"--- gradient health: {name} ({len(g)} logged steps)")
    if not fin.any():
        log("  no finite gradient norms logged")
        return
    log(f"  global norm: median {np.median(g[fin]):.3g}, p99 {np.percentile(g[fin], 99):.3g}, max {g[fin].max():.3g}; "
        f"non-finite {int((~fin).sum())}; clipped {np.mean(g[fin] > grad_clip):.1%}")
    med = {n: np.nanmedian(np.array(v, dtype=float)) if np.isfinite(np.array(v, dtype=float)).any() else float("nan")
           for n, v in st["layers"].items()}
    log("  per-layer median grad norm:", {n: f"{v:.2e}" for n, v in med.items()})
    finite_med = [v for v in med.values() if np.isfinite(v) and v > 0]
    if len(finite_med) >= 2:
        ratio = min(finite_med) / max(finite_med)
        log(f"  smallest/largest layer ratio {ratio:.1e} -> {'OK' if ratio > 1e-4 else 'possible VANISHING gradient'}")
    dead = {n: np.nanmean(np.array(v, dtype=float)) for n, v in st["act_zero"].items() if np.isfinite(np.array(v, dtype=float)).any()}
    log("  mean zero-activation fraction:", {n: f"{v:.2f}" for n, v in dead.items()},
        "-> " + ("OK" if all(v < 0.95 for v in dead.values()) else "some layer is (almost) DEAD"))
    log("  verdict:", "no exploding gradients" if fin.all() and g[fin].max() < 1e3 else "check for EXPLODING gradients")
