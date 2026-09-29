"""Phase 3 evaluation of the full pipeline (detect -> crop -> read) on the held-out LPD test split.

Protocol
--------
* The pipeline runs ONCE per test image at a low detector threshold (0.05); every detection is read and cached, so
  any operating threshold can be evaluated afterwards without re-running the models (threshold sweep).
* Detection: predictions (conf >= t) are matched greedily, most confident first, to ground-truth boxes at IoU >= 0.5.
* Reading: plate_labels.csv gives each image's plate strings but not which box they belong to. For images with one
  plate the pairing is trivial; with several plates, the matched predictions are assigned to the image's strings by
  minimum total edit distance (Hungarian algorithm). Unassigned strings count as missed plates.
* A plate is read correctly only if all 8 characters match exactly.

Metrics: detection P/R/F1; end-to-end P/R/F1 (a correct read of a real plate); recognition accuracy on detected
plates; character error rate (CER); per-position accuracy; image-level accuracy (every plate right, nothing extra).
"""
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .plates import LETTER_LATIN, normalize_plate, plate_tokens, to_latin, edit_distance
from .train_detector import box_iou, read_yolo_boxes


def load_ground_truth(lpd_dir, test_files):
    lpd_dir = Path(lpd_dir)
    texts = {r.image_path: [normalize_plate(t) for t in r.label.split()]
             for r in pd.read_csv(lpd_dir / "plate_labels.csv", dtype=str, keep_default_na=False).itertuples()}
    items = []
    for f in test_files:
        p = lpd_dir / "images" / f
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        items.append({"image": f, "path": p, "W": W, "H": H,
                      "gt_boxes": read_yolo_boxes(lpd_dir / "labels" / f"{Path(f).stem}.txt", W, H),
                      "gt_texts": texts.get(f, [])})
    return items


def run_pipeline(reader, items, low_conf=0.05, log=print):
    """Detect at a low threshold and read every detection once; returns the cache + mean timings."""
    cache, t_det, t_rec, t_tot = [], [], [], []
    for it in tqdm(items, desc="end-to-end on test images", mininterval=10):
        t0 = time.perf_counter()
        n_det0, n_rec0 = len(reader.timings["detect_ms"]), len(reader.timings["recognize_ms"])
        res = reader.predict(it["path"], conf=low_conf)
        t_tot.append((time.perf_counter() - t0) * 1000)
        t_det += reader.timings["detect_ms"][n_det0:]
        t_rec += reader.timings["recognize_ms"][n_rec0:]
        cache.append({"boxes": np.array([r.box for r in res]).reshape(-1, 4), "det_conf": np.array([r.det_conf for r in res]),
                      "texts": [r.text for r in res], "text_conf": np.array([r.text_conf for r in res])})
    timing = {"detect_ms": float(np.mean(t_det[5:] or t_det)), "recognize_ms": float(np.mean(t_rec[5:] or t_rec or [0])),
              "total_ms": float(np.mean(t_tot[5:] or t_tot))}
    log(f"pipeline speed (mean per image, first 5 excluded as warm-up): {timing}")
    return cache, timing


def _assign_texts(pred_texts, gt_texts):
    """Pair predicted strings with ground-truth strings of one image by minimum total edit distance."""
    if not pred_texts or not gt_texts:
        return []
    if len(gt_texts) == 1 and len(pred_texts) == 1:
        return [(0, 0)]
    from scipy.optimize import linear_sum_assignment

    cost = np.array([[edit_distance(plate_tokens(p), plate_tokens(g)) for g in gt_texts] for p in pred_texts])
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows, cols))


def evaluate_at(items, cache, thr, iou_thr=0.5):
    """All metrics at detector threshold `thr` + per-plate and per-image tables."""
    plate_rows, img_rows = [], []
    n_pred = n_gt = n_tp = n_correct = 0
    edits = toks = 0
    pos_ok = np.zeros(8)
    pos_n = 0
    for it, c in zip(items, cache):
        keep = c["det_conf"] >= thr
        pb, pconf, ptext, ptc = c["boxes"][keep], c["det_conf"][keep], [t for t, k in zip(c["texts"], keep) if k], c["text_conf"][keep]
        gb, gtexts = it["gt_boxes"], it["gt_texts"]
        iou = box_iou(pb, gb)
        matched = {}  # pred idx -> gt box idx
        for pi in np.argsort(-pconf):
            if iou.shape[1] == 0:
                break
            cand = [g for g in np.argsort(-iou[pi]) if g not in matched.values() and iou[pi, g] >= iou_thr]
            if cand:
                matched[int(pi)] = int(cand[0])
        m_preds = sorted(matched)
        pairs = _assign_texts([ptext[i] for i in m_preds], gtexts)
        text_of = {m_preds[r]: gtexts[g] for r, g in pairs}
        n_correct_img = 0
        for pi in range(len(pb)):
            gt_text = text_of.get(pi)
            row = {"image": it["image"], "pred_text": ptext[pi], "pred_latin": to_latin(ptext[pi]), "det_conf": float(pconf[pi]),
                   "text_conf": float(ptc[pi]), "box": [round(float(v), 1) for v in pb[pi]], "matched": pi in matched,
                   "iou": float(iou[pi, matched[pi]]) if pi in matched else 0.0, "gt_text": gt_text,
                   "gt_latin": to_latin(gt_text) if gt_text else None}
            if pi in matched and gt_text is not None:
                a, b = plate_tokens(ptext[pi]), plate_tokens(gt_text)
                e = edit_distance(a, b)
                row.update(edit_distance=e, correct=e == 0)
                edits, toks = edits + e, toks + len(b)
                if len(a) == len(b) == 8:
                    pos_ok += np.array([x == y for x, y in zip(a, b)])
                    pos_n += 1
                n_correct_img += e == 0
            else:
                row.update(edit_distance=None, correct=False)
            plate_rows.append(row)
        n_det_gt = len(matched)
        n_pred += len(pb)
        n_gt += max(len(gb), len(gtexts))
        n_tp += n_det_gt
        n_correct += n_correct_img
        n_gt_img = max(len(gb), len(gtexts))
        img_rows.append({"image": it["image"], "n_gt": n_gt_img, "n_pred": len(pb), "detected": n_det_gt,
                         "read_correct": n_correct_img, "false_boxes": len(pb) - n_det_gt,
                         "all_correct": n_correct_img == n_gt_img and len(pb) == n_gt_img})
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0  # noqa: E731
    det_p, det_r = n_tp / max(n_pred, 1), n_tp / max(n_gt, 1)
    e2e_p, e2e_r = n_correct / max(n_pred, 1), n_correct / max(n_gt, 1)
    imgs = pd.DataFrame(img_rows)
    metrics = {
        "threshold": float(thr), "gt_plates": n_gt, "predicted_plates": n_pred, "detected_plates": n_tp,
        "correct_reads": n_correct,
        "det_precision": det_p, "det_recall": det_r, "det_f1": f1(det_p, det_r),
        "e2e_precision": e2e_p, "e2e_recall": e2e_r, "e2e_f1": f1(e2e_p, e2e_r),
        "recognition_acc_on_detected": n_correct / max(n_tp, 1),
        "cer_on_detected": edits / max(toks, 1),
        "char_acc_on_detected": float(pos_ok.sum() / max(pos_n * 8, 1)),
        "image_acc": float(imgs.all_correct.mean()) if len(imgs) else 0.0,
    }
    metrics.update({f"pos{i}_acc": float(pos_ok[i] / max(pos_n, 1)) for i in range(8)})
    return metrics, pd.DataFrame(plate_rows), imgs


def threshold_sweep(items, cache, thresholds=None):
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    return pd.DataFrame([evaluate_at(items, cache, t)[0] for t in thresholds])


# ---------------------------------------------------------------- figures
def plot_sweep(sweep, chosen, trained_thr, save):
    plt.figure()
    for c, lab, st in (("e2e_precision", "end-to-end precision", "-"), ("e2e_recall", "end-to-end recall", "-"),
                       ("e2e_f1", "end-to-end F1", "-"), ("det_recall", "detection recall", "--"),
                       ("det_precision", "detection precision", "--")):
        plt.plot(sweep.threshold, sweep[c], st, marker="o", ms=4, label=lab)
    plt.axvline(chosen, color="k", ls=":", label=f"best F1 threshold {chosen:.2f}")
    plt.axvline(trained_thr, color="gray", ls="-.", label=f"Phase 1 threshold {trained_thr:.2f}")
    plt.title("end-to-end metrics vs detector confidence threshold (test split)")
    plt.xlabel("detector confidence threshold")
    plt.ylabel("score")
    plt.ylim(0, 1.02)
    plt.legend(fontsize=10, loc="lower left")
    save("e2e_threshold_sweep")


def plot_summary_bars(metrics, save):
    keys = [("det_precision", "detection\nprecision"), ("det_recall", "detection\nrecall"), ("e2e_precision", "end-to-end\nprecision"),
            ("e2e_recall", "end-to-end\nrecall"), ("e2e_f1", "end-to-end\nF1"), ("recognition_acc_on_detected", "reading acc.\n(detected)"),
            ("char_acc_on_detected", "char acc.\n(detected)"), ("image_acc", "image\naccuracy")]
    v = [metrics[k] for k, _ in keys]
    plt.figure(figsize=(14, 6.5))
    plt.bar(range(len(v)), v, color=["tab:blue"] * 2 + ["tab:green"] * 3 + ["tab:orange"] * 2 + ["tab:purple"])
    for i, x in enumerate(v):
        plt.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=12)
    plt.xticks(range(len(v)), [l for _, l in keys])
    plt.ylim(0, 1.1)
    plt.title(f"Phase 3 end-to-end results on the test split (threshold {metrics['threshold']:.2f})")
    save("e2e_summary")


def plot_error_breakdown(metrics, plates, save):
    detected = plates[plates.matched & plates.gt_text.notna()]
    one = int((detected.edit_distance == 1).sum())
    more = int((detected.edit_distance >= 2).sum())
    missed = metrics["gt_plates"] - metrics["detected_plates"]
    parts = {"read correctly": metrics["correct_reads"], "1 character wrong": one, "2+ characters wrong": more,
             "not detected": missed, "false box": metrics["predicted_plates"] - metrics["detected_plates"]}
    plt.figure()
    plt.bar(list(parts), list(parts.values()), color=["tab:green", "tab:orange", "tab:red", "tab:gray", "tab:purple"])
    for i, v in enumerate(parts.values()):
        plt.text(i, v, str(v), ha="center", va="bottom", fontsize=12)
    plt.yscale("log")
    plt.ylabel("plates (log scale)")
    plt.title("where the end-to-end errors come from")
    save("e2e_error_breakdown")
    return parts


def plot_position_accuracy(metrics, save):
    v = [metrics[f"pos{i}_acc"] for i in range(8)]
    plt.figure()
    plt.bar(range(8), v, color=["tab:orange" if i == 2 else "tab:blue" for i in range(8)])
    for i, x in enumerate(v):
        plt.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=11)
    plt.xticks(range(8), [f"slot {i}\n{'letter' if i == 2 else 'digit'}" for i in range(8)])
    plt.ylim(0.8, 1.01)
    plt.title("end-to-end accuracy per character position (detected plates)")
    save("e2e_position_accuracy")


def plot_accuracy_by_size(plates, save):
    p = plates[plates.gt_text.notna()].copy()
    p["h"] = [b[3] - b[1] for b in p.box]
    hb, hl = [0, 15, 25, 40, 60, 100, 1e5], ["<15", "15-25", "25-40", "40-60", "60-100", ">100"]
    p["bin"] = pd.cut(p.h, hb, labels=hl)
    acc = p.groupby("bin", observed=False).correct.mean().reindex(hl).fillna(0)
    cnt = p.bin.value_counts().reindex(hl).fillna(0).astype(int)
    plt.figure()
    plt.bar(range(len(hl)), acc.values, color="tab:green")
    for i, v in enumerate(acc.values):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.xticks(range(len(hl)), [f"{l} px\n(n={c})" for l, c in zip(hl, cnt)])
    plt.ylim(0, 1.1)
    plt.ylabel("plates read correctly")
    plt.title("reading accuracy of detected plates by plate height")
    save("e2e_accuracy_by_plate_height")
    return pd.DataFrame({"height_bin": hl, "n_plates": cnt.values, "read_accuracy": acc.values})


def plot_letter_confusion(plates, letter_vocab, save):
    from .plates import parse_plate

    d = plates[plates.matched & plates.gt_text.notna()]
    pairs = [(parse_plate(g), parse_plate(p)) for g, p in zip(d.gt_text, d.pred_text)]
    pairs = [(g[1], p[1]) for g, p in pairs if g and p]
    labels = [c for c in letter_vocab if any(c in pq for pq in pairs)]
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), int)
    for g, p in pairs:
        cm[idx[g], idx[p]] += 1
    plt.figure(figsize=(13, 11))
    plt.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="fraction of the true class")
    for (ti, pi), v in np.ndenumerate(cm):
        if v:
            plt.text(pi, ti, str(v), ha="center", va="center", fontsize=9, color="white" if v / max(cm[ti].sum(), 1) > 0.5 else "black")
    names = [LETTER_LATIN[c] for c in labels]
    plt.xticks(range(len(labels)), names, rotation=70)
    plt.yticks(range(len(labels)), [f"{n} ({cm[i].sum()})" for i, n in enumerate(names)])
    plt.grid(False)
    plt.xlabel("predicted letter")
    plt.ylabel("true letter (count)")
    plt.title("end-to-end letter confusion (detected plates)")
    save("e2e_letter_confusion")


def plot_gallery(items, plates, save, correct=True, n=6, seed=0):
    by_img = {it["image"]: it for it in items}
    sel = plates[plates.matched & plates.gt_text.notna() & (plates.correct == correct)]
    if not len(sel):
        return
    sel = sel.sample(min(n, len(sel)), random_state=seed)
    fig, axes = plt.subplots(len(sel), 2, figsize=(16, 3.2 * len(sel)), gridspec_kw={"width_ratios": [2.2, 1]})
    axes = np.atleast_2d(axes)
    for (a_img, a_crop), row in zip(axes, sel.itertuples()):
        img = cv2.cvtColor(cv2.imread(str(by_img[row.image]["path"])), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = map(int, row.box)
        crop = img[max(0, y1):y2, max(0, x1):x2].copy()  # cut the crop before drawing on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0) if correct else (230, 0, 0), max(3, img.shape[1] // 250))
        a_img.imshow(img)
        a_img.set_title(row.image, fontsize=12)
        a_crop.imshow(crop)
        a_crop.set_title(f"true {row.gt_latin}\npred {row.pred_latin}  ({row.text_conf:.2f})", fontsize=12,
                         color="green" if correct else "red")
        for a in (a_img, a_crop):
            a.axis("off")
    plt.suptitle(f"end-to-end examples: {'correct' if correct else 'wrong'} reads", fontsize=18)
    save(f"e2e_examples_{'correct' if correct else 'wrong'}")


def plot_missed(items, cache, thr, save, n=6):
    """Test images where a ground-truth plate was not detected at the operating threshold."""
    rows = []
    for it, c in zip(items, cache):
        keep = c["det_conf"] >= thr
        iou = box_iou(it["gt_boxes"], c["boxes"][keep])
        for gi in range(len(it["gt_boxes"])):
            if not iou.shape[1] or iou[gi].max() < 0.5:
                rows.append((it, gi))
    if not rows:
        return 0
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for a in axes.ravel():
        a.axis("off")
    for a, (it, gi) in zip(axes.ravel(), rows[:n]):
        img = cv2.cvtColor(cv2.imread(str(it["path"])), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = it["gt_boxes"][gi].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), max(3, img.shape[1] // 250))
        a.imshow(img)
        a.set_title(f"{it['image']}: missed plate {y2 - y1} px high", fontsize=13)
    plt.suptitle("plates the detector missed (orange = ground truth)", fontsize=18)
    save("e2e_missed_plates")
    return len(rows)
