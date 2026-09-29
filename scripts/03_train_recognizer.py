"""Phase 2: train the plate recognizers (pure PyTorch): Model I from scratch, Model II MobileNetV3 fine-tuning.

    python scripts/03_train_recognizer.py --arch both        # full training (GPU recommended, ~45 min on a T4)
    python scripts/03_train_recognizer.py --arch scratch --smoke

Run 01_prepare_data.py first. Every run checkpoints each epoch to work/runs/lpr/<run>/ and resumes automatically.
Writes work/runs/lpr/lpr_<arch>_best.pt (state_dict + vocab) and reports/phase2/ (curves, history CSVs, log).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from platereader import data_lpr as L, recognizer as R, train_recognizer as T  # noqa: E402
from platereader.config import add_common_args, config_from_args  # noqa: E402
from platereader.plates import parse_plate  # noqa: E402
from platereader.utils import FigureSaver, RunLog, pick_device, set_seed  # noqa: E402


def load_splits(cfg):
    P = cfg.paths
    vocab = (P.splits_dir / "letter_vocab.txt").read_text(encoding="utf-8").split("\n")
    dfs = {}
    for s in ("train", "val", "test"):
        df = pd.read_csv(P.splits_dir / f"lpr_{s}_split.csv", dtype=str, encoding="utf-8").rename(columns={"label": "text"})
        df["letter"] = [parse_plate(t)[1] for t in df.text]
        n = cfg["lpr"]["smoke_n"]
        dfs[s] = df.sample(n=min(len(df), n[s]), random_state=cfg.seed).reset_index(drop=True) if n else df
    return dfs, vocab


def save_checkpoint(model, arch, vocab, backbone, path):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "arch": arch,
                "n_letters": len(vocab), "letter_vocab": vocab, "input_size": list(R.INPUT_SIZE), "backbone": backbone}, path)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--arch", default="both", choices=["scratch", "pretrained", "both"])
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    P, C = cfg.paths, cfg["lpr"]
    set_seed(cfg.seed)
    out = P.reports("recognizer")
    log, save = RunLog(out / "run_log.txt"), FigureSaver(out / "figures")
    device = pick_device(args.device)
    workers = args.workers if args.workers is not None else (4 if device.type == "cuda" else 0)

    dfs, vocab = load_splits(cfg)
    enc = L.Encoder(vocab)
    img_dir = P.lpr_dir / "detections"
    ds = {s: L.PLPRDataset(d, img_dir, enc, augment=L.TrainAugment(C["crop_jitter"]) if s == "train" else None)
          for s, d in dfs.items()}
    loaders = {s: L.make_loader(d, C["batch_size"], s == "train", workers, device.type == "cuda", cfg.seed) for s, d in ds.items()}
    log(f"device={device} | samples:", {s: len(d) for s, d in ds.items()})
    criterion = T.PlateLoss(T.letter_class_weights(dfs["train"], vocab, C["letter_weight_power"], C["max_letter_weight"]),
                            C["label_smoothing"]).to(device)
    common = dict(device=device, criterion=criterion, grad_clip=C["grad_clip"], pct_start=C["pct_start"],
                  log_every=C["log_every"], log=log)
    runs = P.runs_dir / "lpr"
    runs.mkdir(parents=True, exist_ok=True)

    if args.arch in ("scratch", "both"):
        set_seed(cfg.seed)
        sc = C["scratch"]
        model = R.ScratchLPRNet(len(vocab))
        log("Model I (scratch) params:", T.count_params(model))
        hist, _ = T.fit(model, runs / f"scratch_{P.tag}", loaders, T.param_groups(model, sc["lr"], sc["weight_decay"]),
                        sc["epochs"], sc["patience"], **common)
        save_checkpoint(model, "scratch", vocab, None, runs / "lpr_scratch_best.pt")
        T.save_history_csv(hist, out, "scratch")
        T.plot_history(hist, "Model I: scratch CNN", "scratch", save, C["grad_clip"])
        T.grad_health_report(hist, "scratch", C["grad_clip"], log)

    if args.arch in ("pretrained", "both"):
        set_seed(cfg.seed)
        pc = C["pretrained"]
        model = R.PretrainedLPRNet(len(vocab), pretrained=True, backbone=pc["backbone"])
        model.set_backbone_trainable(False)
        log("Model II stage 1 (frozen backbone) trainable params:", T.count_params(model)[1])
        h1, best1 = T.fit(model, runs / f"pretrained_stage1_{P.tag}", loaders,
                          T.param_groups(model.head, pc["stage1_lr"], pc["weight_decay"]),
                          pc["stage1_epochs"], pc["stage1_epochs"], **common)
        model.set_backbone_trainable(True)
        groups = T.param_groups(model.features, pc["backbone_lr"], pc["weight_decay"]) + \
            T.param_groups(model.head, pc["head_lr"], pc["weight_decay"])
        h2, _ = T.fit(model, runs / f"pretrained_stage2_{P.tag}", loaders, groups, pc["epochs"], pc["patience"],
                      init_state=torch.load(best1, map_location="cpu", weights_only=True)["state_dict"], **common)
        hist = T.concat_histories(h1, h2)
        save_checkpoint(model, "pretrained", vocab, pc["backbone"], runs / "lpr_pretrained_best.pt")
        T.save_history_csv(hist, out, "pretrained")
        T.plot_history(hist, "Model II: MobileNetV3 (frozen -> fine-tuned)", "pretrained", save, C["grad_clip"])
        T.grad_health_report(hist, "pretrained", C["grad_clip"], log)
    log(f"done: checkpoints in {runs}, results in {out}")


if __name__ == "__main__":
    main()
