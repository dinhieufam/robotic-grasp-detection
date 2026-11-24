"""Evaluate a trained model (best.pt by validation accuracy) with the rectangle metric.

    python scripts/eval_model.py --config configs/cornell_rgbd.yaml --subsets val test

Writes runs/<name>/eval.json (accuracy, 95% CI, failure split, timing) and
runs/<name>/per_image_<subset>.csv. Test subsets are for final numbers only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grasp.config import load_config, repo_path  # noqa: E402
from grasp.data.dataset import CHANNELS, CachedGraspData, GraspDataset  # noqa: E402
from grasp.evaluate import evaluate, write_rows  # noqa: E402
from grasp.models.gcnet import build_model  # noqa: E402
from grasp.postprocess import maps_to_grasps  # noqa: E402


def load_model(cfg, device, which="best"):
    model = build_model(cfg["model"], len(CHANNELS[cfg["data"]["channels"]])).to(device)
    ck = torch.load(repo_path(os.path.join("checkpoints", cfg["name"], f"{which}.pt")), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    return model.eval(), ck["epoch"]


@torch.no_grad()
def time_inference(model, x, ecfg, w_scale, device, n=200, warmup=30):
    """Median ms per single image: forward + post-processing (GPU synchronised)."""
    x = x[None].to(device)
    ts = []
    for i in range(warmup + n):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        g = maps_to_grasps(model(x), ecfg["q_sigma"], ecfg["ang_sigma"], ecfg["w_sigma"], w_scale, ecfg["jaw_ratio"])
        g.cpu()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            ts.append(1000 * (time.perf_counter() - t0))
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subsets", nargs="+", default=["val"])
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    args = ap.parse_args()
    cfg = load_config(repo_path(args.config))
    dcfg = load_config(repo_path(cfg["data_config"]))
    ecfg, lab = cfg["eval"], dcfg["label_maps"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(cfg, device, args.checkpoint)
    data = CachedGraspData(repo_path(dcfg["paths"]["cache"]))
    run_dir = repo_path(os.path.join("runs", cfg["name"]))
    out_path = os.path.join(run_dir, "eval.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for subset in args.subsets:
        ds = GraspDataset(data, repo_path(cfg["data"]["split"]), subset, cfg["data"]["channels"], label_cfg=lab)
        row_of = {k: i for i, k in enumerate(ds.rows)}

        @torch.no_grad()
        def predict(k):
            x = ds[row_of[k]]["x"][None].to(device)
            return maps_to_grasps(model(x), ecfg["q_sigma"], ecfg["ang_sigma"], ecfg["w_sigma"], lab["w_scale"],
                                  ecfg["jaw_ratio"])[0].cpu().numpy()

        ids = [int(data.img_ids[k]) for k in ds.rows]
        summary, rows = evaluate(predict, data, ids, ecfg["angle_thresh_deg"], ecfg["iou_thresh"])
        summary.pop("ms_per_image_mean"); summary.pop("ms_per_image_median")  # includes data loading; see timing
        summary["checkpoint"], summary["epoch"] = args.checkpoint, epoch
        results[subset] = summary
        write_rows(os.path.join(run_dir, f"per_image_{subset}.csv"), rows)
        print(f"{cfg['name']} {subset}: acc {summary['accuracy']:.3f} [{summary['ci95'][0]:.3f}, "
              f"{summary['ci95'][1]:.3f}] n={summary['n']} (angle fails {summary['failures_angle']}, "
              f"IoU-only {summary['failures_iou_only']}) epoch {epoch}")

    med, p95 = time_inference(model, ds[0]["x"], ecfg, lab["w_scale"], device)
    results["timing"] = {"ms_median": med, "ms_p95": p95, "batch": 1,
                         "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    print(f"inference {med:.2f} ms median (p95 {p95:.2f}) on {results['timing']['device']}")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
