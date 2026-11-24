"""Dataset statistics + figures for the reports (Phase 1).

    python scripts/make_data_figures.py --config configs/data_cornell.yaml [--example 100]

Writes to runs/data_stats/:
    split_stats.json / split_stats.tex   images, objects, rectangles per split (both split types)
    hist_angle_width.pdf/.png            grasp angle + opening width distributions per split
    example_pipeline.pdf/.png            raw -> cleaned -> label maps (+ one augmented view)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grasp.config import load_config, repo_path  # noqa: E402
from grasp.data.cornell import list_images, load_corners, load_depth_from_pcd  # noqa: E402
from grasp.data.dataset import CachedGraspData  # noqa: E402
from grasp.data.labels import draw_label_maps, is_degenerate, rect_to_corners  # noqa: E402
from grasp.data.transforms import Augmenter  # noqa: E402

SPLIT_COLOURS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}  # categorical slots 1-3
INK, MUTED = "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def split_stats(data: CachedGraspData, splits_dir: str):
    out = {}
    for kind in ("objectwise", "imagewise"):
        with open(os.path.join(splits_dir, f"cornell_{kind}.json")) as f:
            s = json.load(f)
        out[kind] = {}
        for sub in ("train", "val", "test"):
            rows = [data.index[i] for i in s[sub]]
            n_rects = np.array([len(data.rects(k)) for k in rows])
            out[kind][sub] = {"images": len(rows), "objects": int(len(set(data.obj_ids[rows]))),
                              "rects": int(n_rects.sum()), "rects_per_image_mean": float(n_rects.mean()),
                              "rects_per_image_std": float(n_rects.std())}
    return out


def stats_table_tex(stats) -> str:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Split type & Subset & Images & Objects & Rectangles & Rect./image \\", r"\midrule"]
    for kind, label in (("objectwise", "Object-wise"), ("imagewise", "Image-wise")):
        for j, sub in enumerate(("train", "val", "test")):
            s = stats[kind][sub]
            lines.append(f"{label if j == 0 else ''} & {sub} & {s['images']} & {s['objects']} & {s['rects']} & "
                         f"{s['rects_per_image_mean']:.1f} $\\pm$ {s['rects_per_image_std']:.1f} \\\\")
        lines.append(r"\midrule" if kind == "objectwise" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def histograms(data, splits_dir, out):
    with open(os.path.join(splits_dir, "cornell_objectwise.json")) as f:
        s = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 1.9))
    bins_a, bins_w = np.linspace(-90, 90, 37), np.linspace(0, 120, 41)
    for sub, col in SPLIT_COLOURS.items():
        R = np.concatenate([data.rects(data.index[i]) for i in s[sub]])
        axes[0].hist(np.rad2deg(R[:, 2]), bins=bins_a, density=True, histtype="step", lw=2, color=col, label=sub)
        axes[1].hist(R[:, 3], bins=bins_w, density=True, histtype="step", lw=2, color=col, label=sub)
    axes[0].set(xlabel=r"grasp angle $\theta$ (deg)", ylabel="density", xticks=[-90, -45, 0, 45, 90])
    axes[1].set(xlabel="gripper opening $w$ (px, 224 scale)")
    axes[1].legend(frameon=False, title="object-wise split", title_fontsize=7)
    for a in axes:
        a.grid(axis="y", color="#e6e5e0", lw=0.6)
        a.set_axisbelow(True)
    save(fig, out, "hist_angle_width")


def draw_rects(ax, rects, colour="#eda100", lw=1.0):
    for r in rects:
        c = rect_to_corners(r)
        ax.add_patch(Polygon(c, closed=True, fill=False, ec=colour, lw=lw * 0.6))
        for a, b in ((1, 2), (3, 0)):  # the two jaw plates (perpendicular to the opening axis)
            ax.plot(c[[a, b], 0], c[[a, b], 1], color="#d03b3b", lw=lw * 1.5)


def example_figure(data, cfg, img_id, out):
    P = {k: repo_path(v) for k, v in cfg["paths"].items()}
    prefix = list_images(P["raw"])[img_id]
    k = data.index[img_id]
    rgb_raw = cv2.cvtColor(cv2.imread(prefix + "r.png"), cv2.COLOR_BGR2RGB)
    depth_raw = load_depth_from_pcd(prefix + ".txt")
    corners = [c for c in load_corners(prefix + "cpos.txt") if not is_degenerate(c)]
    x0, y0 = np.load(P["cache"])["crop_xy"][k]
    S = cfg["crop"]["size_px"]

    rgb, depth, rects = data.rgb[k], data.depth[k], data.rects(k)
    L = draw_label_maps(rects, depth.shape, cfg["label_maps"]["shrink"], cfg["label_maps"]["w_scale"])
    aug = Augmenter(cfg["augment"])
    rgb_a, depth_a, rects_a = aug(rgb, depth, rects, np.random.default_rng(3))

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.7))
    ax = axes[0, 0]; ax.imshow(rgb_raw)
    for c in corners:
        ax.add_patch(Polygon(c, closed=True, fill=False, ec="#eda100", lw=0.6))
    ax.add_patch(Rectangle((x0, y0), S, S, fill=False, ec="white", lw=1.2, ls="--"))
    ax.set_title("raw RGB 640×480 + crop", fontsize=8)
    ax = axes[0, 1]; d = np.ma.masked_where(depth_raw == 0, depth_raw)
    cm = plt.get_cmap("viridis").copy(); cm.set_bad("#d03b3b")
    ax.imshow(d, cmap=cm); ax.set_title(f"raw depth (holes red)", fontsize=8)
    ax = axes[0, 2]; ax.imshow(rgb); draw_rects(ax, rects); ax.set_title("cleaned RGB 224 + labels", fontsize=8)
    if str(np.load(P["cache"])["depth_mode"]) == "table":
        ax = axes[0, 3]; im = ax.imshow(depth * 100, cmap="Blues", vmin=0, vmax=max(1.0, np.percentile(depth, 99.5) * 100))
        ax.set_title("height above table (cm)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    else:
        ax = axes[0, 3]; ax.imshow(depth, cmap="viridis"); ax.set_title("inpainted, mean-sub. depth", fontsize=8)
    ax = axes[1, 0]; ax.imshow(L[0], cmap="Blues", vmin=0, vmax=1); ax.set_title("Q map", fontsize=8)
    ang = np.ma.masked_where(L[0] == 0, 0.5 * np.arctan2(L[2], L[1]))
    cm2 = plt.get_cmap("twilight").copy(); cm2.set_bad("white")
    ax = axes[1, 1]; im = ax.imshow(np.rad2deg(ang), cmap=cm2, vmin=-90, vmax=90); ax.set_title(r"angle $\theta$ (deg)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[-90, 0, 90])
    cm3 = plt.get_cmap("Blues").copy(); cm3.set_bad("white")
    ax = axes[1, 2]; im = ax.imshow(np.ma.masked_where(L[0] == 0, L[3]), cmap=cm3, vmin=0, vmax=1); ax.set_title(r"width $W=w/150$", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[0, 0.5, 1])
    ax = axes[1, 3]; ax.imshow(rgb_a); draw_rects(ax, rects_a); ax.set_title("augmented sample", fontsize=8)
    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout(pad=0.3)
    save(fig, out, "example_pipeline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_cornell.yaml")
    ap.add_argument("--example", type=int, default=None, help="image id for the pipeline figure")
    args = ap.parse_args()
    cfg = load_config(repo_path(args.config))
    P = {k: repo_path(v) for k, v in cfg["paths"].items()}
    os.makedirs(P["stats_dir"], exist_ok=True)
    data = CachedGraspData(P["cache"])

    stats = split_stats(data, P["splits_dir"])
    with open(os.path.join(P["stats_dir"], "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(P["stats_dir"], "split_stats.tex"), "w") as f:
        f.write(stats_table_tex(stats))
    print(json.dumps(stats["objectwise"], indent=1))
    histograms(data, P["splits_dir"], P["stats_dir"])
    example_figure(data, cfg, args.example if args.example is not None else int(data.img_ids[0]), P["stats_dir"])
    print("figures ->", P["stats_dir"])


if __name__ == "__main__":
    main()
