"""Phase 1: train and evaluate the YOLO26 licence-plate detector (Ultralytics)."""
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .utils import free_gpu


def run_state(run_dir):
    """'new' | 'resume' | 'finished' (Ultralytics sets epoch=-1 in last.pt when training completed)."""
    last = Path(run_dir) / "weights" / "last.pt"
    if not last.exists():
        return "new"
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    return "finished" if ckpt.get("epoch", -1) == -1 else "resume"


def losses_finite(run_dir):
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df.columns = df.columns.str.strip()
    return bool(np.isfinite(df[[c for c in df.columns if "loss" in c]].to_numpy(dtype=float)).all())


def train_detector(weights, data_yaml, project, name, train_cfg, device, workers=0, seed=42, log=print):
    """Train (or resume / skip if finished). `device` may be a list of GPUs (DDP); on failure retries on one GPU."""
    from ultralytics import YOLO

    run_dir = Path(project) / name

    def _train(dev):
        n_dev = len(dev) if isinstance(dev, list) else 1
        state = run_state(run_dir)
        if state == "finished":
            log(f"{run_dir} already finished -> skipping training")
        elif state == "resume":
            log(f"resuming from {run_dir / 'weights' / 'last.pt'}")
            YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True, device=dev, workers=workers)
        else:
            YOLO(str(weights)).train(data=str(data_yaml), project=str(project), name=name, exist_ok=True, device=dev,
                                     workers=workers, seed=seed, **{**train_cfg, "batch": train_cfg["batch"] * n_dev})

    try:
        _train(device)
        ok = losses_finite(run_dir)
    except Exception as e:
        if isinstance(device, list):
            log(f"multi-GPU (DDP) training failed: {e!r} -> retrying on a single GPU")
            device = 0
            _train(device)
            ok = losses_finite(run_dir)
        elif device == "mps":
            log("YOLO training on MPS failed:", repr(e))
            ok = False
        else:
            raise
    if not ok:
        if device == "mps":
            log("NaN/failed on MPS -> retrying on CPU")
            shutil.rmtree(run_dir, ignore_errors=True)
            _train("cpu")
        else:
            raise RuntimeError("YOLO training produced non-finite losses")
    best = run_dir / "weights" / "best.pt"
    assert best.exists(), best
    return run_dir, best


def predict_in_chunks(model, paths, imgsz, device, conf=0.25, chunk=16):
    """YOLO predict on many paths, `chunk` at a time. (A single predict() on a long list batches ALL images at once
    in Ultralytics 8.4 -> CUDA out-of-memory.) Returns [(xyxy [N,4], conf [N], (H, W)), ...]."""
    out, paths = [], [str(p) for p in paths]
    for i in range(0, len(paths), chunk):
        for r in model.predict(paths[i: i + chunk], imgsz=imgsz, conf=conf, device=device, verbose=False):
            out.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.orig_shape))
    return out


def det_metrics(m):
    return {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
            "mAP50-95": float(m.box.map), "fitness": float(m.fitness)}


def best_f1_conf(val_metrics, lo=0.05, hi=0.9, default=0.25):
    try:
        f1 = np.asarray(val_metrics.box.f1_curve).mean(0)
        return float(np.clip(np.asarray(val_metrics.box.px)[f1.argmax()], lo, hi))
    except Exception:
        return default


def plot_training_curves(results_csv, save):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    cols = list(df.columns)
    ep = np.asarray(df["epoch"]) if "epoch" in cols else np.arange(1, len(df) + 1)
    for k in sorted({c.split("/", 1)[1] for c in cols if "loss" in c and "/" in c}):  # YOLO26: box, cls, l1 (no dfl)
        plt.figure()
        for split, color in (("train", "tab:blue"), ("val", "tab:orange")):
            if f"{split}/{k}" in cols:
                plt.plot(ep, df[f"{split}/{k}"], marker="o", ms=4, color=color, label=split)
        plt.title(f"YOLO26 {k} per epoch")
        plt.xlabel("epoch")
        plt.ylabel(k)
        plt.legend()
        save(f"lpd_{k}")
    metric_cols = {"precision": next((c for c in cols if "precision" in c), None),
                   "recall": next((c for c in cols if "recall" in c), None),
                   "mAP50": next((c for c in cols if "mAP50(" in c), None),
                   "mAP50-95": next((c for c in cols if "mAP50-95" in c), None)}
    plt.figure()
    for name, c in metric_cols.items():
        if c:
            plt.plot(ep, df[c], marker="o", ms=4, label=name)
    plt.title("YOLO26 validation metrics per epoch")
    plt.xlabel("epoch")
    plt.ylim(0, 1.02)
    plt.legend()
    save("lpd_val_metrics_overview")
    return df


def box_iou(a, b):
    """IoU matrix between xyxy boxes a [N,4] and b [M,4]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def read_yolo_boxes(label_path, W, H):
    lines = Path(label_path).read_text().splitlines() if Path(label_path).exists() else []
    return np.array([[(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]
                     for _, cx, cy, bw, bh in (map(float, l.split()) for l in lines if l.strip())]).reshape(-1, 4)


def detector_error_analysis(model, image_paths, label_dir, imgsz, device, conf_thr, out_dir, save, iou_thr=0.5, log=print):
    """Match every test plate to its best prediction; IoU/confidence/recall-by-size figures, CSVs, hardest images."""
    free_gpu()
    gt_rows, img_rows, fp_conf, cache = [], [], [], {}
    for p, (pb, pc, (H, W)) in zip(image_paths, predict_in_chunks(model, image_paths, imgsz, device, conf=0.01)):
        gt = read_yolo_boxes(Path(label_dir) / f"{Path(p).stem}.txt", W, H)
        keep = pc >= conf_thr
        pb, kc = pb[keep], pc[keep]
        iou = box_iou(gt, pb)
        used = set()
        for gi in range(len(gt)):
            cand = [j for j in np.argsort(-iou[gi]) if j not in used and iou[gi, j] >= iou_thr] if iou.shape[1] else []
            j = cand[0] if cand else -1
            if j >= 0:
                used.add(j)
            gt_rows.append({"image": Path(p).name, "gt_w_px": gt[gi, 2] - gt[gi, 0], "gt_h_px": gt[gi, 3] - gt[gi, 1],
                            "detected": j >= 0, "best_iou": float(iou[gi].max()) if iou.shape[1] else 0.0,
                            "conf": float(kc[j]) if j >= 0 else np.nan})
        fp = [j for j in range(len(kc)) if j not in used]
        fp_conf += [float(kc[j]) for j in fp]
        img_rows.append({"image": Path(p).name, "n_gt": len(gt), "n_pred": len(kc), "tp": len(used),
                         "fn": len(gt) - len(used), "fp": len(fp)})
        cache[Path(p).name] = (gt, pb, kc, p)
    boxes, images = pd.DataFrame(gt_rows), pd.DataFrame(img_rows)
    boxes.to_csv(Path(out_dir) / "lpd_test_boxes.csv", index=False)
    images.to_csv(Path(out_dir) / "lpd_test_images.csv", index=False)
    n_det, n_gt = int(boxes.detected.sum()), len(boxes)
    bad = images[(images.fn > 0) | (images.fp > 0)]
    log(f"LPD test @ conf {conf_thr:.3f}: {n_det}/{n_gt} plates detected (recall {n_det / max(n_gt, 1):.4f}), "
        f"{len(fp_conf)} false boxes, {len(bad)}/{len(images)} images with at least one error")

    plt.figure()
    plt.hist(boxes.best_iou[boxes.detected], bins=25, range=(0.5, 1), color="tab:green")
    plt.title("test: IoU of detected plates with ground truth")
    plt.xlabel("IoU")
    plt.ylabel("plates")
    save("lpd_test_iou_hist")

    plt.figure()
    plt.hist([boxes.conf.dropna(), fp_conf], bins=20, range=(0, 1), label=["true positive", "false positive"],
             color=["tab:green", "tab:red"])
    plt.axvline(conf_thr, color="k", ls="--", label=f"threshold {conf_thr:.2f}")
    plt.title("test: detection confidence, true vs false")
    plt.xlabel("confidence")
    plt.ylabel("detections")
    plt.legend()
    save("lpd_test_confidence_hist")

    hb, hl = [0, 15, 25, 40, 60, 100, 1e5], ["<15", "15-25", "25-40", "40-60", "60-100", ">100"]
    hc = pd.cut(boxes.gt_h_px, hb, labels=hl)
    rec = boxes.detected.groupby(hc, observed=False).mean().reindex(hl).fillna(0)
    cnt = hc.value_counts().reindex(hl).fillna(0).astype(int)
    plt.figure()
    plt.bar(range(len(hl)), rec.values, color="tab:blue")
    for i, v in enumerate(rec.values):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.xticks(range(len(hl)), [f"{l} px\n(n={c})" for l, c in zip(hl, cnt)])
    plt.title("test: recall by plate height (native resolution)")
    plt.ylabel("recall")
    plt.ylim(0, 1.1)
    save("lpd_test_recall_by_plate_height")
    pd.DataFrame({"height_bin": hl, "n_plates": cnt.values, "recall": rec.values}).to_csv(
        Path(out_dir) / "lpd_recall_by_plate_height.csv", index=False)

    worst = bad.assign(err=bad.fn + bad.fp).sort_values("err", ascending=False).head(6)
    if len(worst):
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        for a in axes.ravel():
            a.axis("off")
        for a, row in zip(axes.ravel(), worst.itertuples()):
            gt, pb, kc, path = cache[row.image]
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            t = max(2, img.shape[1] // 300)
            for x1, y1, x2, y2 in gt.astype(int):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), t)
            for (x1, y1, x2, y2), c in zip(pb.astype(int), kc):
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), t)
                cv2.putText(img, f"{c:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, img.shape[1] / 1200, (255, 0, 0), t)
            a.imshow(img)
            a.set_title(f"{row.image}: missed {row.fn}, false {row.fp}", fontsize=13)
        plt.suptitle("hardest test images (green = ground truth, red = prediction)", fontsize=18)
        save("lpd_test_worst_images")
    free_gpu()
    return boxes, images


def show_ultralytics_plots(save_dirs, save):
    for split, d in save_dirs.items():
        for pat in ("confusion_matrix_normalized.png", "*PR_curve.png", "*F1_curve.png"):
            for p in sorted(Path(d).glob(pat)):
                plt.figure(figsize=(12, 9))
                plt.imshow(Image.open(p))
                plt.title(f"{split}: {p.stem}")
                plt.axis("off")
                save(f"lpd_{split}_{p.stem}")
