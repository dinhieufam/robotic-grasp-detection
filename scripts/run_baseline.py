"""Evaluate the PCA heuristic baseline.

    python scripts/run_baseline.py --config configs/baseline.yaml

Writes runs/<name>/: config.yaml, metrics.json, per_image_<split>_<subset>.csv
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grasp.baseline import predict  # noqa: E402
from grasp.config import load_config, repo_path  # noqa: E402
from grasp.data.dataset import CachedGraspData  # noqa: E402
from grasp.evaluate import evaluate, write_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()
    cfg = load_config(repo_path(args.config))
    dcfg = load_config(repo_path(cfg["data_config"]))
    data = CachedGraspData(repo_path(dcfg["paths"]["cache"]))
    out = repo_path(os.path.join("runs", cfg["name"]))
    os.makedirs(out, exist_ok=True)
    shutil.copy(repo_path(args.config), os.path.join(out, "config.yaml"))

    bcfg, ecfg = cfg["baseline"], cfg["eval"]
    metrics = {}
    for split_name, split_path in cfg["splits"].items():
        with open(repo_path(split_path)) as f:
            split = json.load(f)
        for subset in cfg["subsets"]:
            summary, rows = evaluate(lambda k: predict(data.depth[k], bcfg)["grasp"], data, split[subset],
                                     ecfg["angle_thresh_deg"], ecfg["iou_thresh"])
            metrics[f"{split_name}/{subset}"] = summary
            write_rows(os.path.join(out, f"per_image_{split_name}_{subset}.csv"), rows)
            print(f"{split_name:>10}/{subset:<5} acc {summary['accuracy']:.3f} [{summary['ci95'][0]:.3f}, {summary['ci95'][1]:.3f}]  (n={summary['n']}, "
                  f"angle fails {summary['failures_angle']}, IoU-only fails {summary['failures_iou_only']}, "
                  f"{summary['ms_per_image_mean']:.2f} ms/img)")
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
