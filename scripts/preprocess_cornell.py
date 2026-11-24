"""Preprocess Cornell (proposal steps 1-5) into one .npz cache + split files.

    python scripts/preprocess_cornell.py --config configs/data_cornell.yaml [--workers 6]

Outputs
    data/cornell/cache/cornell_224.npz      rgb, depth, holes, rects (+ offsets), ids, crops
    data/splits/cornell_{objectwise,imagewise}.json
    runs/data_stats/preprocess_log.json     how many images/rectangles were dropped at each step
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grasp.config import load_config, repo_path  # noqa: E402
from grasp.data.cornell import list_images, load_object_ids, preprocess_image  # noqa: E402
from grasp.data.splits import image_wise_split, object_wise_split  # noqa: E402

_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg


def _work(item):
    img_id, prefix = item
    return preprocess_image(prefix, img_id, _CFG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_cornell.yaml")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    args = ap.parse_args()

    cfg = load_config(repo_path(args.config))
    P = {k: repo_path(v) for k, v in cfg["paths"].items()}
    images = list_images(P["raw"])
    objects = load_object_ids(P["object_ids"])
    items = list(images.items())[: args.limit or None]
    print(f"{len(items)} raw images, {len({objects[i][0] for i, _ in items})} objects")

    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        results = pool.map(_work, items, chunksize=4)
    print(f"processed in {time.time() - t0:.0f}s")

    log = {"images_raw": len(results), "objects_raw": len({objects[r.img_id][0] for r in results}),
           "dropped_images": {}, "rects": {"raw": 0, "degenerate": 0, "outside_crop": 0}}
    for r in results:
        log["rects"]["raw"] += r.counts.get("rects_raw", 0)
        log["rects"]["degenerate"] += r.counts.get("rects_degenerate", 0)
        log["rects"]["outside_crop"] += r.counts.get("rects_outside", 0)
        if r.status != "ok":
            log["dropped_images"].setdefault(r.status, []).append(r.img_id)
    ok = [r for r in results if r.status == "ok"]
    log["images_kept"] = len(ok)
    log["objects_kept"] = len({objects[r.img_id][0] for r in ok})
    log["rects"]["kept"] = int(sum(len(r.rects) for r in ok))
    ti = np.array([r.table_inliers for r in ok], dtype=float)
    if np.isfinite(ti).any():
        log["table_plane_inlier_frac"] = {"median": float(np.nanmedian(ti)), "min": float(np.nanmin(ti)),
                                          "min_img_id": int(ok[int(np.nanargmin(ti))].img_id)}
    log["missing_depth_frac_kept"] = {"mean": float(np.mean([r.missing_frac for r in ok])),
                                     "max": float(np.max([r.missing_frac for r in ok]))}

    img_ids = np.array([r.img_id for r in ok])
    obj_ids = np.array([objects[i][0] for i in img_ids])
    offsets = np.concatenate([[0], np.cumsum([len(r.rects) for r in ok])]).astype(np.int64)
    os.makedirs(os.path.dirname(P["cache"]), exist_ok=True)
    np.savez(P["cache"],
             rgb=np.stack([r.rgb for r in ok]), depth=np.stack([r.depth for r in ok]),
             holes=np.stack([r.holes for r in ok]), rects=np.concatenate([r.rects for r in ok]),
             rect_offsets=offsets, img_ids=img_ids, obj_ids=obj_ids,
             obj_names=np.array([objects[i][1] for i in img_ids]),
             crop_xy=np.array([r.crop_xy for r in ok]), missing_frac=np.array([r.missing_frac for r in ok]),
             table_inliers=np.array([r.table_inliers for r in ok]), depth_mode=np.array(cfg["depth"]["normalise"]))
    print("cache ->", P["cache"])

    fr, seed = cfg["split"]["fractions"], cfg["seed"]
    os.makedirs(P["splits_dir"], exist_ok=True)
    for name, split in [("objectwise", object_wise_split(img_ids, obj_ids, fr, seed)),
                        ("imagewise", image_wise_split(img_ids, fr, seed))]:
        path = os.path.join(P["splits_dir"], f"cornell_{name}.json")
        with open(path, "w") as f:
            json.dump({"seed": seed, "fractions": fr, **split}, f)
        log[f"split_{name}"] = {k: len(v) for k, v in split.items()}
        print(name, log[f"split_{name}"], "->", path)

    os.makedirs(P["stats_dir"], exist_ok=True)
    with open(os.path.join(P["stats_dir"], "preprocess_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    shutil.copy(repo_path(args.config), os.path.join(P["stats_dir"], "config.yaml"))
    print(json.dumps({k: (v if not isinstance(v, dict) or k != "dropped_images" else {s: len(l) for s, l in v.items()})
                      for k, v in log.items()}, indent=2))


if __name__ == "__main__":
    main()
