"""Train the grasp network.

    python -m grasp.train --config configs/cornell_rgbd.yaml [--resume]

Writes runs/<name>/{config.yaml, log.csv, metrics.json} and
checkpoints/<name>/{last.pt, best.pt}. A checkpoint is saved every epoch;
--resume continues from last.pt. The test split is never used here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import load_config, repo_path
from .data.dataset import CHANNELS, CachedGraspData, GraspDataset, collate
from .metrics import match_grasp
from .models.gcnet import build_model
from .postprocess import maps_to_grasps

LOG_FIELDS = ["epoch", "lr", "train_loss", "train_q", "train_cos", "train_sin", "train_w",
              "val_loss", "val_acc", "time_s"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grasp_loss(pred, y):
    """Sum of Smooth-L1 losses of the four maps (proposal). Returns total and per-map values."""
    parts = [F.smooth_l1_loss(pred[:, i], y[:, i]) for i in range(4)]
    return sum(parts), parts


def make_loaders(cfg, dcfg):
    data = CachedGraspData(repo_path(dcfg["paths"]["cache"]))
    split = repo_path(cfg["data"]["split"])
    lab = dcfg["label_maps"]
    ch = cfg["data"]["channels"]
    aug = dcfg["augment"] if cfg["data"].get("augment", True) else None
    train = GraspDataset(data, split, "train", ch, augment=aug, label_cfg=lab, seed=cfg["seed"])
    val = GraspDataset(data, split, "val", ch, augment=None, label_cfg=lab)
    nw = cfg["train"].get("num_workers", 6)
    val_loader = DataLoader(val, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=nw,
                            collate_fn=collate)
    return data, train, val_loader


@torch.no_grad()
def run_eval(model, loader, device, ecfg, lab):
    """Validation loss + rectangle-metric accuracy."""
    model.eval()
    losses, correct = [], []
    for b in loader:
        x, y = b["x"].to(device), b["y"].to(device)
        out = model(x)
        losses.append(grasp_loss(out, y)[0].item() * len(x))
        grasps = maps_to_grasps(out, ecfg["q_sigma"], ecfg["ang_sigma"], ecfg["w_sigma"], lab["w_scale"],
                                ecfg["jaw_ratio"]).cpu().numpy()
        for g, r in zip(grasps, b["rects"]):
            correct.append(match_grasp(g, r.numpy(), ecfg["angle_thresh_deg"], ecfg["iou_thresh"])["correct"])
    return float(np.sum(losses) / len(correct)), float(np.mean(correct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override (debugging)")
    args = ap.parse_args()

    cfg = load_config(repo_path(args.config))
    dcfg = load_config(repo_path(cfg["data_config"]))
    tcfg, ecfg = cfg["train"], cfg["eval"]
    epochs = args.epochs or tcfg["epochs"]
    run_dir, ckpt_dir = repo_path(os.path.join("runs", cfg["name"])), repo_path(os.path.join("checkpoints", cfg["name"]))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, train_set, val_loader = make_loaders(cfg, dcfg)
    model = build_model(cfg["model"], len(CHANNELS[cfg["data"]["channels"]])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    sched = None
    if tcfg.get("scheduler") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_params = sum(p.numel() for p in model.parameters())

    start, best, log_rows, train_time = 1, {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0}, [], 0.0
    last_path, best_path, log_path = (os.path.join(ckpt_dir, "last.pt"), os.path.join(ckpt_dir, "best.pt"),
                                      os.path.join(run_dir, "log.csv"))
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if sched and ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        start, best, train_time = ck["epoch"] + 1, ck["best"], ck.get("train_time", 0.0)
        torch.set_rng_state(ck["torch_rng"].cpu())
        with open(log_path) as f:
            log_rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= ck["epoch"]]
        print(f"resumed from epoch {ck['epoch']} (best val acc {best['val_acc']:.3f} @ {best['epoch']})")
    else:
        shutil.copy(repo_path(args.config), os.path.join(run_dir, "config.yaml"))
    print(f"{cfg['name']}: {n_params:,} params, {len(train_set)} train imgs, device {device}")

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        train_set.set_epoch(epoch)
        loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                            num_workers=tcfg.get("num_workers", 6), collate_fn=collate,
                            generator=torch.Generator().manual_seed(cfg["seed"] * 1000 + epoch))
        sums, n = np.zeros(5), 0
        for b in loader:
            x, y = b["x"].to(device, non_blocking=True), b["y"].to(device, non_blocking=True)
            loss, parts = grasp_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums += np.array([loss.item()] + [p.item() for p in parts]) * len(x)
            n += len(x)
        if sched:
            sched.step()
        val_loss, val_acc = run_eval(model, val_loader, device, ecfg, dcfg["label_maps"])
        dt = time.time() - t0
        train_time += dt
        row = dict(zip(LOG_FIELDS, [epoch, opt.param_groups[0]["lr"], *(sums / n), val_loss, val_acc, dt]))
        log_rows.append(row)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS); w.writeheader(); w.writerows(log_rows)

        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
                    "epoch": epoch, "best": best, "torch_rng": torch.get_rng_state(), "train_time": train_time,
                    "cfg": cfg}, last_path)
        print(f"ep {epoch:3d}  loss {row['train_loss']:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  "
              f"(best {best['val_acc']:.3f} @ {best['epoch']})  {dt:.1f}s", flush=True)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"params": n_params, "epochs": epochs, "best_val_acc": best["val_acc"], "best_epoch": best["epoch"],
                   "best_val_loss": best["val_loss"], "train_time_s": train_time,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, f, indent=2)


if __name__ == "__main__":
    main()
