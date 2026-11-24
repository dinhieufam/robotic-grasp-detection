"""Primary-model figures and tables (Phase 3).

    python scripts/make_model_figures.py --config configs/cornell_rgbd.yaml [--baseline configs/baseline.yaml]

Writes to runs/<name>/:
    learning_curves.pdf/.png    train/val loss + val accuracy per epoch (baseline val accuracy for reference)
    layer_table.tex             layer, output shape, parameters
    architecture.pdf/.png       block diagram
    qualitative.pdf/.png        test images: input, Q / angle / width maps, model vs baseline grasp
    results_table.tex           baseline vs model accuracy (object-wise + image-wise test, 95% CI) + timing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from eval_model import load_model  # noqa: E402
from grasp.baseline import predict as baseline_predict  # noqa: E402
from grasp.config import load_config, repo_path  # noqa: E402
from grasp.data.dataset import CachedGraspData, GraspDataset  # noqa: E402
from grasp.data.labels import rect_to_corners  # noqa: E402
from grasp.metrics import match_grasp  # noqa: E402
from grasp.postprocess import blur, maps_to_grasps  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, INK, MUTED, GRID = "#d03b3b", "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def read_log(run_dir):
    with open(os.path.join(run_dir, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


# ---------------------------------------------------------------- learning curves
def learning_curves(run_dir, out, baseline_val=None, best_epoch=None, figsize=(6.4, 2.1), name="learning_curves",
                    short=False):
    L = read_log(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    ax.plot(L["epoch"], L["train_loss"], color=BLUE, lw=2, label="train")
    ax.plot(L["epoch"], L["val_loss"], color=ORANGE, lw=2, label="validation")
    ax.set(xlabel="epoch", ylabel="loss" if short else "loss (sum of 4 Smooth-L1)", yscale="log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(L["epoch"], 100 * L["val_acc"], color=ORANGE, lw=2, label="model (validation)")
    if baseline_val is not None:
        ax.axhline(100 * baseline_val, color=MUTED, lw=1.2, ls="--")
        ax.text(L["epoch"][-1], 100 * baseline_val - 1.5, "baseline (val)" if short else "PCA baseline (val)", ha="right",
                va="top", color=MUTED, fontsize=plt.rcParams["font.size"] - 1)
    if best_epoch:
        i = int(best_epoch) - 1
        ax.plot(L["epoch"][i], 100 * L["val_acc"][i], "o", ms=6, color=ORANGE, mec="white", mew=1.5)
        ax.annotate(f"best: {100 * L['val_acc'][i]:.1f}% @ {int(best_epoch)}", (L["epoch"][i], 100 * L["val_acc"][i]),
                    xytext=(-4 if short else 0, 6), textcoords="offset points", ha="right" if short else "center",
                    fontsize=plt.rcParams["font.size"] - 1, color=INK)
    ax.set(xlabel="epoch", ylabel="val. accuracy (%)" if short else "grasp accuracy (%)",
           ylim=(min(40, 100 * L["val_acc"].min() - 5), 100))
    for a in axes:
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, name)


# ---------------------------------------------------------------- layer table
def layer_table(model, in_ch, out):
    rows, hooks = [], []
    names = ["enc1", "enc2", "enc3", "res", "dec1", "dec2", "dec3", "head_q", "head_cos", "head_sin", "head_w"]
    desc = {"enc1": "Conv 5$\\times$5, s2 + BN + ReLU", "enc2": "Conv 3$\\times$3, s2 + BN + ReLU",
            "enc3": "Conv 3$\\times$3, s2 + BN + ReLU", "res": "5 $\\times$ residual block (2 Conv 3$\\times$3 + BN)",
            "dec1": "ConvT 4$\\times$4, s2 + BN + ReLU", "dec2": "concat enc2 $\\to$ ConvT 4$\\times$4, s2",
            "dec3": "concat enc1 $\\to$ ConvT 4$\\times$4, s2", "head_q": "1$\\times$1 Conv, sigmoid ($Q$)",
            "head_cos": "1$\\times$1 Conv, tanh ($\\cos2\\theta$)", "head_sin": "1$\\times$1 Conv, tanh ($\\sin2\\theta$)",
            "head_w": "1$\\times$1 Conv, sigmoid ($W$)"}
    shapes = {}
    for n in names:
        hooks.append(getattr(model, n).register_forward_hook(lambda m, i, o, n=n: shapes.__setitem__(n, tuple(o.shape[1:]))))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_ch, 224, 224, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    total = 0
    lines = [r"\begin{tabular}{llrr}", r"\toprule", r"Block & Operation & Output & Params \\", r"\midrule",
             f"input & RGB-D & {in_ch}$\\times$224$\\times$224 & -- \\\\"]
    for n in names:
        p = sum(q.numel() for q in getattr(model, n).parameters())
        total += p
        s = shapes[n]
        tex_name = n.replace("_", "\\_")
        lines.append(f"{tex_name} & {desc[n]} & {s[0]}$\\times${s[1]}$\\times${s[2]} & {p:,} \\\\")
    lines += [r"\midrule", f"\\multicolumn{{3}}{{l}}{{Total trainable parameters}} & {total:,} \\\\", r"\bottomrule",
              r"\end{tabular}"]
    with open(os.path.join(out, "layer_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return total


# ---------------------------------------------------------------- architecture diagram
def architecture(out, n_params, figsize=(7.2, 2.5), fs=1.0, header=True, name="architecture"):
    """``fs`` scales every font size (use < 1 when the figure is drawn smaller)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100); ax.set_ylim(0.5 if header else 2.5, 39 if header else 33.5); ax.axis("off")
    # (label, sublabel, x, height, colour)
    blocks = [("input", "4×224²\nRGB-D", 2, 24, "#d9d8d2"),
              ("enc1", "32×112²", 13, 18, BLUE), ("enc2", "64×56²", 23, 13, BLUE), ("enc3", "128×28²", 33, 9, BLUE),
              ("5× Res", "128×28²", 44, 9, "#184f95"),
              ("dec1", "64×56²", 56, 13, AQUA), ("dec2", "32×112²", 66, 18, AQUA), ("dec3", "32×224²", 76, 24, AQUA)]
    ybase = 6
    centres = []
    for lab, sub, x, h, col in blocks:
        w = 7.5 if lab != "5× Res" else 9
        ax.add_patch(FancyBboxPatch((x, ybase), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=col, ec="none",
                                    alpha=0.9 if col != "#d9d8d2" else 1))
        ax.text(x + w / 2, ybase + h / 2, lab, ha="center", va="center", color="white" if col != "#d9d8d2" else INK,
                fontsize=7.5 * fs, weight="bold")
        ax.text(x + w / 2, ybase - 1.2, sub, ha="center", va="top", color=MUTED, fontsize=6.5 * fs)
        centres.append((x, w, h))
    for (x0, w0, h0), (x1, _, h1) in zip(centres[:-1], centres[1:]):
        ax.add_artist(FancyArrowPatch((x0 + w0 + 0.4, ybase + 4), (x1 - 0.4, ybase + 4), arrowstyle="-|>",
                                      mutation_scale=7, color=MUTED, lw=0.8))
    # skip connections enc2->dec2 input, enc1->dec3 input
    for (src, dst) in ((2, 6), (1, 7)):
        xs, ws, hs = centres[src]; xd, _, _ = centres[dst]
        ax.add_artist(FancyArrowPatch((xs + ws / 2, ybase + hs + 0.6), (xd - 1.0, ybase + centres[dst][2] * 0.75),
                                      connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=7,
                                      color=ORANGE, lw=1.0, ls="--"))
    ax.text(46, 27.5, "U-Net skips (concatenate)", color=ORANGE, fontsize=6.5 * fs, ha="center", va="center")
    # heads
    heads = [("Q", "sigmoid"), ("cos 2θ", "tanh"), ("sin 2θ", "tanh"), ("W", "sigmoid")]
    xd, wd, hd = centres[-1]
    for i, (h, act) in enumerate(heads):
        y = ybase + hd - 3 - i * 6.2
        ax.add_patch(FancyBboxPatch((88, y), 9, 4.2, boxstyle="round,pad=0.2,rounding_size=0.6", fc="white", ec=INK, lw=0.7))
        ax.text(92.5, y + 2.1, f"{h}\n{act}", ha="center", va="center", fontsize=6 * fs, color=INK, linespacing=0.9)
        ax.add_artist(FancyArrowPatch((xd + wd + 0.4, ybase + hd / 2), (87.6, y + 2.1), arrowstyle="-|>",
                                      mutation_scale=6, color=MUTED, lw=0.6))
    ax.text(92.5, 1.0, "1×1 conv heads\n224×224 each", ha="center", va="bottom", fontsize=6 * fs, color=MUTED)
    if header:
        ax.text(2, 38.5, f"Encoder: strided Conv–BN–ReLU   ·   bottleneck: residual blocks   ·   decoder: ConvT–BN–ReLU   ·   "
                       f"{n_params / 1e6:.2f} M parameters", fontsize=6.8 * fs, color=INK, va="top")
    save(fig, out, name)


# ---------------------------------------------------------------- qualitative
def draw_grasp(ax, r, colour, lw=1.0, jaws=True):
    c = rect_to_corners(r)
    ax.add_patch(Polygon(c, closed=True, fill=False, ec=colour, lw=lw * (0.6 if jaws else 1)))
    if jaws:
        for a, b in ((1, 2), (3, 0)):
            ax.plot(c[[a, b], 0], c[[a, b], 1], color=colour, lw=lw * 2.2, solid_capstyle="butt")


def qualitative(cfg, dcfg, bcfg, model, device, out, n_rows=4):
    ecfg, lab = cfg["eval"], dcfg["label_maps"]
    data = CachedGraspData(repo_path(dcfg["paths"]["cache"]))
    ds = GraspDataset(data, repo_path(cfg["data"]["split"]), "test", cfg["data"]["channels"], label_cfg=lab)
    recs = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]; k = ds.rows[i]
            o = model(s["x"][None].to(device))
            g = maps_to_grasps(o, ecfg["q_sigma"], ecfg["ang_sigma"], ecfg["w_sigma"], lab["w_scale"], ecfg["jaw_ratio"])[0].cpu().numpy()
            gb = baseline_predict(data.depth[k], bcfg)["grasp"]
            recs.append({"i": i, "k": k, "g": g, "gb": gb, "out": o[0].cpu(),
                         "m": match_grasp(g, data.rects(k))["correct"], "b": match_grasp(gb, data.rects(k))["correct"]})
    rng = np.random.default_rng(0)
    pick = lambda c: [c[j] for j in rng.permutation(len(c))]  # noqa: E731
    fixes = pick([r for r in recs if r["m"] and not r["b"]])
    both = pick([r for r in recs if r["m"] and r["b"]])
    fails = pick([r for r in recs if not r["m"]])
    chosen = (fixes[:2] + both[:1] + fails[:1]) if len(fixes) >= 2 else (both[:3] + fails[:1])
    chosen = chosen[:n_rows]

    cols = ["RGB + grasps", "height (cm)", "Q (smoothed)", "angle θ (deg)", "width W"]
    fig, axes = plt.subplots(len(chosen), 5, figsize=(7.2, 1.55 * len(chosen) + 0.3))
    for r_i, r in enumerate(chosen):
        k, o = r["k"], r["out"]
        ax = axes[r_i, 0]; ax.imshow(data.rgb[k])
        for gt in data.rects(k):
            draw_grasp(ax, gt, YELLOW, 0.6, jaws=False)
        draw_grasp(ax, r["gb"], MUTED, 0.8)
        draw_grasp(ax, r["g"], RED, 1.0)
        tag = ("model ✓" if r["m"] else "model ✗") + " · " + ("baseline ✓" if r["b"] else "baseline ✗")
        ax.set_ylabel(f"{str(data.obj_names[k]).replace('_', ' ')[:16]}\n{tag}", fontsize=6.5, color=INK)
        axes[r_i, 1].imshow(data.depth[k] * 100, cmap="Blues", vmin=0, vmax=max(1, np.percentile(data.depth[k], 99.5) * 100))
        q = blur(o[None, 0:1], ecfg["q_sigma"])[0, 0].numpy()
        axes[r_i, 2].imshow(q, cmap="Blues", vmin=0, vmax=1)
        axes[r_i, 2].plot(r["g"][0], r["g"][1], "+", color=RED, ms=7, mew=1.5)
        ang = np.rad2deg(0.5 * np.arctan2(o[2].numpy(), o[1].numpy()))
        axes[r_i, 3].imshow(np.ma.masked_where(q < 0.1, ang), cmap="twilight", vmin=-90, vmax=90)
        axes[r_i, 4].imshow(np.ma.masked_where(q < 0.1, o[3].numpy()), cmap="Blues", vmin=0, vmax=0.8)
        for c_i, a in enumerate(axes[r_i]):
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if r_i == 0:
                a.set_title(cols[c_i], fontsize=7.5, color=INK)
    handles = [plt.Line2D([], [], color=RED, lw=2, label="model grasp"),
               plt.Line2D([], [], color=MUTED, lw=2, label="baseline grasp"),
               plt.Line2D([], [], color=YELLOW, lw=1, label="labelled grasps")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.4, w_pad=0.3)
    save(fig, out, "qualitative")
    return {"test_n": len(recs), "model_correct": int(sum(r["m"] for r in recs)),
            "baseline_correct": int(sum(r["b"] for r in recs)),
            "model_fixes_baseline": len(fixes), "model_breaks_baseline": int(sum(r["b"] and not r["m"] for r in recs))}


def results_table(out, rows):
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Model & Object-wise test & Image-wise test \\", r"\midrule"]
    for name, ow, iw in rows:
        f = lambda s: "--" if s is None else f"{100 * s['accuracy']:.1f}\\% ({100 * s['ci95'][0]:.0f}--{100 * s['ci95'][1]:.0f})"  # noqa: E731
        lines.append(f"{name} & {f(ow)} & {f(iw)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "results_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cornell_rgbd.yaml")
    ap.add_argument("--imagewise-config", default="configs/cornell_rgbd_imagewise.yaml")
    ap.add_argument("--baseline", default="configs/baseline.yaml")
    args = ap.parse_args()
    cfg = load_config(repo_path(args.config))
    dcfg = load_config(repo_path(cfg["data_config"]))
    bcfg = load_config(repo_path(args.baseline))
    run_dir = repo_path(os.path.join("runs", cfg["name"]))
    bmetrics = json.load(open(repo_path(os.path.join("runs", bcfg["name"], "metrics.json"))))
    metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
    ev = json.load(open(os.path.join(run_dir, "eval.json")))

    learning_curves(run_dir, run_dir, bmetrics["objectwise/val"]["accuracy"], metrics["best_epoch"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(cfg, device)
    in_ch = model.enc1[0].in_channels
    n_params = layer_table(model, in_ch, run_dir)
    architecture(run_dir, n_params)
    q = qualitative(cfg, dcfg, bcfg["baseline"], model, device, run_dir)
    print("test comparison (object-wise):", q)

    iw_path = repo_path(os.path.join("runs", load_config(repo_path(args.imagewise_config))["name"], "eval.json"))
    iw = json.load(open(iw_path)).get("test") if os.path.exists(iw_path) else None
    results_table(run_dir, [("PCA heuristic baseline", bmetrics["objectwise/test"], bmetrics["imagewise/test"]),
                            ("Our network (RGB-D)", ev.get("test"), iw)])
    with open(os.path.join(run_dir, "figure_stats.json"), "w") as f:
        json.dump({"objectwise_test_vs_baseline": q, "params": n_params, "timing": ev.get("timing")}, f, indent=2)
    print("figures ->", run_dir)


if __name__ == "__main__":
    main()
