"""Baseline figures + table for the reports (Phase 2).

    python scripts/make_baseline_figures.py --config configs/baseline.yaml [--diagram-id 462]

Reads runs/<name>/ (written by run_baseline.py) and writes there:
    baseline_diagram.pdf/.png       the 5 heuristic steps on one image
    baseline_qualitative.pdf/.png   successes and failures on the object-wise test set
    baseline_table.tex              accuracy (95% CI) per split
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grasp.baseline import predict  # noqa: E402
from grasp.config import load_config, repo_path  # noqa: E402
from grasp.data.dataset import CachedGraspData  # noqa: E402
from grasp.data.labels import rect_to_corners  # noqa: E402

GT, PRED, AXIS = "#eda100", "#d03b3b", "#2a78d6"
INK, MUTED = "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def draw_grasp(ax, r, colour, lw=1.0, jaws=True):
    c = rect_to_corners(r)
    ax.add_patch(Polygon(c, closed=True, fill=False, ec=colour, lw=lw * (0.6 if jaws else 1)))
    if jaws:
        for a, b in ((1, 2), (3, 0)):
            ax.plot(c[[a, b], 0], c[[a, b], 1], color=colour, lw=lw * 2, solid_capstyle="butt")


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def diagram(data, bcfg, img_id, out, figsize=(7.2, 1.9), titles=None, name="baseline_diagram", title_fs=7):
    k = data.index[img_id]
    h, rgb, gts = data.depth[k], data.rgb[k], data.rects(k)
    res = predict(h, bcfg)
    g, mask = res["grasp"], res["mask"]
    above = h > res["table"] + bcfg["height_thresh"]
    lab, _ = ndimage.label(above, structure=np.ones((3, 3)))

    fig, axes = plt.subplots(1, 5, figsize=figsize)
    titles = titles or ["① height map;\ntable = border median", "② > 1 cm above table", "③ keep largest component\nnot touching border",
              "④ centroid + PCA axes", "⑤ grasp across long axis\nw = extent + 10 px"]
    axes[0].imshow(h * 100, cmap="Blues", vmin=0, vmax=max(1, np.percentile(h, 99) * 100))
    rest = np.zeros((*h.shape, 3)) + 1
    rest[above] = [0.75, 0.75, 0.75]
    axes[1].imshow(rest)
    sel = rest.copy(); sel[mask] = [0.16, 0.47, 0.84]
    axes[2].imshow(sel)
    axes[3].imshow(sel)
    ys, xs = np.nonzero(mask)
    cx, cy = g[0], g[1]
    la = res["long_axis"]
    for ang, ln, col in ((la, 1.0, INK), (la + np.pi / 2, 0.6, PRED)):
        L = 0.5 * ln * max(np.ptp(xs), np.ptp(ys))
        axes[3].plot([cx - L * np.cos(ang), cx + L * np.cos(ang)], [cy + L * np.sin(ang), cy - L * np.sin(ang)],
                     color=col, lw=1.4)
    axes[3].plot(cx, cy, "o", ms=4, color=INK, mec="white", mew=0.8)
    axes[4].imshow(rgb)
    for r in gts:
        draw_grasp(axes[4], r, GT, 0.8, jaws=False)
    draw_grasp(axes[4], g, PRED, 1.2)
    for a, t in zip(axes, titles):
        a.set_title(t, fontsize=title_fs, color=INK, loc="left")
        clean(a)
    fig.tight_layout(w_pad=1.6)
    for a, b in zip(axes[:-1], axes[1:]):
        p0 = a.get_position(); p1 = b.get_position()
        fig.add_artist(FancyArrowPatch(((p0.x1 + 0.002), (p0.y0 + p0.y1) / 2), ((p1.x0 - 0.002), (p1.y0 + p1.y1) / 2),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=0.8))
    save(fig, out, name)


def qualitative(data, bcfg, rows, out, n_each=4, seed=0):
    rng = np.random.default_rng(seed)
    ok = [r for r in rows if r["correct"] == "True"]
    bad = [r for r in rows if r["correct"] == "False"]
    succ = [ok[i] for i in rng.choice(len(ok), n_each, replace=False)]
    ang = sorted([r for r in bad if float(r["min_angle_diff_deg"]) > 30], key=lambda r: -float(r["min_angle_diff_deg"]))
    iou = sorted([r for r in bad if float(r["min_angle_diff_deg"]) <= 30], key=lambda r: float(r["best_iou_angle_ok"]))
    fails = (ang[: n_each // 2] + iou[: n_each - min(len(ang), n_each // 2)])[:n_each]

    fig, axes = plt.subplots(2, n_each, figsize=(7.2, 3.9))
    for row, items, label in ((0, succ, "success"), (1, fails, "failure")):
        for j, r in enumerate(items):
            k = data.index[int(r["img_id"])]
            ax = axes[row, j]
            ax.imshow(data.rgb[k])
            mask = predict(data.depth[k], bcfg)["mask"]
            ax.contour(mask, levels=[0.5], colors=[AXIS], linewidths=0.6)
            for gt in data.rects(k):
                draw_grasp(ax, gt, GT, 0.7, jaws=False)
            draw_grasp(ax, [float(r[c]) for c in ("x", "y", "theta", "w", "h")], PRED, 1.1)
            why = "" if label == "success" else (" · angle" if float(r["min_angle_diff_deg"]) > 30 else " · IoU")
            ax.set_title(f"{r['obj_name'].replace('_', ' ')[:18]}\nIoU {float(r['best_iou_angle_ok']):.2f}, "
                         f"Δθ {float(r['min_angle_diff_deg']):.0f}°{why}", fontsize=7, color=INK)
            clean(ax)
            # zoom on the object mask + all labelled grasps
            ys, xs = np.nonzero(mask)
            pts = np.concatenate([np.stack([xs, ys], 1).reshape(-1, 2)] + [rect_to_corners(gt) for gt in data.rects(k)])
            lo, hi = pts.min(0), pts.max(0)
            c = (lo + hi) / 2; half = min(112, max(45, 0.55 * max(hi - lo)))
            ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] + half, c[1] - half)
        axes[row, 0].set_ylabel(label, fontsize=8, color=INK)
    handles = [plt.Line2D([], [], color=PRED, lw=2, label="baseline grasp"),
               plt.Line2D([], [], color=GT, lw=1, label="labelled grasps"),
               plt.Line2D([], [], color=AXIS, lw=1, label="object mask")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "baseline_qualitative")


def table(metrics, out):
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Split & Subset & $n$ & Accuracy (95\% CI) & Angle / IoU fails \\", r"\midrule"]
    for key, s in metrics.items():
        split, sub = key.split("/")
        lines.append(f"{'Object-wise' if split == 'objectwise' else 'Image-wise'} & {sub} & {s['n']} & "
                     f"{100 * s['accuracy']:.1f}\\% ({100 * s['ci95'][0]:.0f}--{100 * s['ci95'][1]:.0f}) & "
                     f"{s['failures_angle']} / {s['failures_iou_only']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "baseline_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--diagram-id", type=int, default=462)
    args = ap.parse_args()
    cfg = load_config(repo_path(args.config))
    dcfg = load_config(repo_path(cfg["data_config"]))
    data = CachedGraspData(repo_path(dcfg["paths"]["cache"]))
    out = repo_path(os.path.join("runs", cfg["name"]))
    with open(os.path.join(out, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(out, "per_image_objectwise_test.csv")) as f:
        rows = list(csv.DictReader(f))
    diagram(data, cfg["baseline"], args.diagram_id, out)
    qualitative(data, cfg["baseline"], rows, out)
    table(metrics, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
