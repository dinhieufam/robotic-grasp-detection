"""Build every figure, table and number used in the progress report from runs/.

    CUDA_VISIBLE_DEVICES= python scripts/make_progress_report_assets.py

Writes docs/progress_report/{numbers.tex, fig/*.pdf, tab/*.tex}. Nothing in the
report is typed by hand: the text uses the macros in numbers.tex.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from eval_model import load_model  # noqa: E402
import make_baseline_figures as mbf  # noqa: E402
import make_model_figures as mmf  # noqa: E402
from grasp.baseline import predict as baseline_predict  # noqa: E402
from grasp.config import load_config, repo_path  # noqa: E402
from grasp.data.cornell import list_images, load_corners, load_depth_from_pcd, is_degenerate  # noqa: E402
from grasp.data.dataset import CachedGraspData, GraspDataset  # noqa: E402
from grasp.data.labels import draw_label_maps, rect_to_corners  # noqa: E402
from grasp.data.transforms import Augmenter  # noqa: E402
from grasp.metrics import match_grasp  # noqa: E402
from grasp.postprocess import blur, maps_to_grasps  # noqa: E402

OUT = repo_path("docs/progress_report")
FIG, TAB = os.path.join(OUT, "fig"), os.path.join(OUT, "tab")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, INK, MUTED = "#d03b3b", "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.dpi": 300, "axes.titlesize": 7.5})


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_grasp(ax, r, colour, lw=1.0, jaws=True):
    c = rect_to_corners(r)
    ax.add_patch(Polygon(c, closed=True, fill=False, ec=colour, lw=lw * (0.6 if jaws else 1)))
    if jaws:
        for a, b in ((1, 2), (3, 0)):
            ax.plot(c[[a, b], 0], c[[a, b], 1], color=colour, lw=lw * 2.2, solid_capstyle="butt")


def zoom(ax, pts, minhalf=40):
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    half = min(112, max(minhalf, 0.6 * max(hi - lo)))
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] + half, c[1] - half)


def model_outputs(model, ds, device, ecfg, lab):
    recs = []
    with torch.no_grad():
        for i in range(len(ds)):
            s, k = ds[i], ds.rows[i]
            o = model(s["x"][None].to(device))
            g = maps_to_grasps(o, ecfg["q_sigma"], ecfg["ang_sigma"], ecfg["w_sigma"], lab["w_scale"],
                               ecfg["jaw_ratio"])[0].cpu().numpy()
            recs.append({"k": k, "g": g, "out": o[0].cpu(), "m": match_grasp(g, ds.data.rects(k))})
    return recs


# ------------------------------------------------------------------ figures
def teaser(data, rec, ecfg):
    """Fig. 1: input (RGB + height) -> CNN -> Q / angle / width maps -> grasp."""
    k, o, g = rec["k"], rec["out"], rec["g"]
    q = blur(o[None, 0:1], ecfg["q_sigma"])[0, 0].numpy()
    ang = np.rad2deg(0.5 * np.arctan2(o[2].numpy(), o[1].numpy()))
    fig = plt.figure(figsize=(7.0, 1.6))
    xs = [0.0, 0.125, 0.315, 0.43, 0.555, 0.68, 0.84]
    w = 0.115
    panels = [("RGB", data.rgb[k], {}), ("height above table", data.depth[k] * 100,
                                                {"cmap": "Blues", "vmin": 0, "vmax": max(1, np.percentile(data.depth[k], 99.5) * 100)}),
              None, ("quality Q", q, {"cmap": "Blues", "vmin": 0, "vmax": 1}),
              ("angle θ", np.ma.masked_where(q < 0.1, ang), {"cmap": "twilight", "vmin": -90, "vmax": 90}),
              ("width W", np.ma.masked_where(q < 0.1, o[3].numpy()), {"cmap": "Blues", "vmin": 0, "vmax": 0.8}),
              ("best grasp", data.rgb[k], {})]
    axes = []
    for x, p in zip(xs, panels):
        if p is None:
            ax = fig.add_axes([x - 0.02, 0.08, 0.14, 0.72]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.add_patch(FancyBboxPatch((0.1, 0.2), 0.8, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08", fc=BLUE, ec="none"))
            ax.text(0.5, 0.5, "our CNN\n1.8 M params\n3 ms / image", ha="center", va="center", color="white", fontsize=7)
            axes.append(ax); continue
        ax = fig.add_axes([x, 0.08, w, 0.72]); ax.imshow(p[1], **p[2]); ax.set_title(p[0], color=INK, fontsize=7); clean(ax)
        axes.append(ax)
    for x0, x1, t in ((xs[0], xs[1] + w, "input (224×224, 4 channels)"), (xs[3], xs[5] + w, "output: 4 per-pixel maps"),
                      (xs[6], xs[6] + w, "output grasp")):
        fig.text((x0 + x1) / 2, 0.97, t, ha="center", va="top", fontsize=7.5, color=INK, weight="bold")
    # zoom all image panels on the object
    ys_, xs_ = np.nonzero(q > 0.1)
    pts = np.concatenate([np.stack([xs_, ys_], 1)] + [rect_to_corners(gt) for gt in data.rects(k)])
    for a in [axes[i] for i in (0, 1, 3, 4, 5, 6)]:
        zoom(a, pts, 50)
    axes[3].plot(g[0], g[1], "+", color=RED, ms=6, mew=1.4)
    for gt in data.rects(k):
        draw_grasp(axes[6], gt, YELLOW, 0.5, jaws=False)
    draw_grasp(axes[6], g, RED, 0.9)
    for a, b in ((1, 2), (2, 3), (5, 6)):
        p0, p1 = axes[a].get_position(), axes[b].get_position()
        x0 = p0.x1 + 0.004 if a != 2 else p0.x0 + 0.13
        x1 = p1.x0 - 0.004 if b != 2 else p1.x0 + 0.01
        fig.add_artist(FancyArrowPatch((x0, 0.44), (x1, 0.44), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=8, color=MUTED, lw=0.9))
    save(fig, "teaser")


def data_example(data, dcfg, img_id):
    """Fig. 2: raw -> cleaned -> label maps -> augmented, one row."""
    P = {k: repo_path(v) for k, v in dcfg["paths"].items()}
    prefix = list_images(P["raw"])[img_id]
    k = data.index[img_id]
    rgb_raw = plt.imread(prefix + "r.png")
    depth_raw = load_depth_from_pcd(prefix + ".txt")
    corners = [c for c in load_corners(prefix + "cpos.txt") if not is_degenerate(c)]
    x0, y0 = np.load(P["cache"])["crop_xy"][k]
    S = dcfg["crop"]["size_px"]
    rgb, h, rects = data.rgb[k], data.depth[k], data.rects(k)
    L = draw_label_maps(rects, h.shape, dcfg["label_maps"]["shrink"], dcfg["label_maps"]["w_scale"])
    rgb_a, _, rects_a = Augmenter(dcfg["augment"])(rgb, h, rects, np.random.default_rng(3))

    fig, axes = plt.subplots(1, 7, figsize=(7.0, 1.3), gridspec_kw={"width_ratios": [4 / 3, 4 / 3, 1, 1, 1, 1, 1]})
    ax = axes[0]; ax.imshow(rgb_raw)
    for c in corners:
        ax.add_patch(Polygon(c, closed=True, fill=False, ec=YELLOW, lw=0.5))
    ax.add_patch(Rectangle((x0, y0), S, S, fill=False, ec="white", lw=0.9, ls="--")); ax.set_title("(a) raw RGB")
    cm = plt.get_cmap("viridis").copy(); cm.set_bad(RED)
    axes[1].imshow(np.ma.masked_where(depth_raw == 0, depth_raw), cmap=cm); axes[1].set_title("(b) raw depth")
    axes[2].imshow(rgb)
    for r in rects:
        draw_grasp(axes[2], r, RED, 0.6)
    axes[2].set_title("(c) RGB crop")
    axes[3].imshow(h * 100, cmap="Blues", vmin=0, vmax=max(1, np.percentile(h, 99.5) * 100)); axes[3].set_title("(d) height")
    axes[4].imshow(L[0], cmap="Blues", vmin=0, vmax=1); axes[4].set_title("(e) Q target")
    axes[5].imshow(np.ma.masked_where(L[0] == 0, np.rad2deg(0.5 * np.arctan2(L[2], L[1]))), cmap="twilight", vmin=-90, vmax=90)
    axes[5].set_title("(f) θ target")
    axes[6].imshow(rgb_a)
    for r in rects_a:
        draw_grasp(axes[6], r, RED, 0.6)
    axes[6].set_title("(g) augmented")
    for a in axes:
        clean(a)
    fig.tight_layout(w_pad=0.2)
    save(fig, "data_example")


def baseline_examples(data, bcfg, rows, n_succ=2, n_fail=2):
    """Fig. 4: baseline successes and failures (object-wise test), one row."""
    rng = np.random.default_rng(0)
    ok = [r for r in rows if r["correct"] == "True"]
    bad = [r for r in rows if r["correct"] == "False"]
    succ = [ok[i] for i in rng.choice(len(ok), n_succ, replace=False)]
    ang = sorted([r for r in bad if float(r["min_angle_diff_deg"]) > 30], key=lambda r: -float(r["min_angle_diff_deg"]))
    iou = sorted([r for r in bad if float(r["min_angle_diff_deg"]) <= 30], key=lambda r: float(r["best_iou_angle_ok"]))
    fails = ang[:1] + iou[:n_fail - 1]
    items = [(r, True) for r in succ] + [(r, False) for r in fails]
    fig, axes = plt.subplots(1, len(items), figsize=(5.6, 1.55))
    for ax, (r, good) in zip(axes, items):
        k = data.index[int(r["img_id"])]
        mask = baseline_predict(data.depth[k], bcfg)["mask"]
        ax.imshow(data.rgb[k]); ax.contour(mask, levels=[0.5], colors=[BLUE], linewidths=0.5)
        for gt in data.rects(k):
            draw_grasp(ax, gt, YELLOW, 0.5, jaws=False)
        g = [float(r[c]) for c in ("x", "y", "theta", "w", "h")]
        draw_grasp(ax, g, RED, 0.9)
        ys, xs = np.nonzero(mask)
        zoom(ax, np.concatenate([np.stack([xs, ys], 1)] + [rect_to_corners(gt) for gt in data.rects(k)]))
        why = "✓ correct" if good else ("✗ angle" if float(r["min_angle_diff_deg"]) > 30 else "✗ IoU")
        ax.set_title(f"{r['obj_name'].replace('_', ' ')[:14]}\n{why}", color=INK, fontsize=7)
        clean(ax)
    save(fig, "baseline_examples")


def model_examples(data, bcfg, recs, ecfg):
    """Fig. 6: two rows -- a thin object the baseline misses, and a model failure."""
    rng = np.random.default_rng(0)
    for r in recs:
        r["gb"] = baseline_predict(data.depth[r["k"]], bcfg)["grasp"]
        r["b"] = match_grasp(r["gb"], data.rects(r["k"]))["correct"]
    fixes = [r for r in recs if r["m"]["correct"] and not r["b"]]
    fails = [r for r in recs if not r["m"]["correct"]]
    chosen = [fixes[i] for i in rng.permutation(len(fixes))[:2]] + [fails[i] for i in rng.permutation(len(fails))[:1]]
    fig, axes = plt.subplots(1, 2 * len(chosen), figsize=(7.0, 1.6))
    for i, r in enumerate(chosen):
        k, o = r["k"], r["out"]
        q = blur(o[None, 0:1], ecfg["q_sigma"])[0, 0].numpy()
        ax, axq = axes[2 * i], axes[2 * i + 1]
        ax.imshow(data.rgb[k])
        for gt in data.rects(k):
            draw_grasp(ax, gt, YELLOW, 0.5, jaws=False)
        draw_grasp(ax, r["gb"], MUTED, 0.7)
        draw_grasp(ax, r["g"], RED, 0.9)
        axq.imshow(q, cmap="Blues", vmin=0, vmax=1); axq.plot(r["g"][0], r["g"][1], "+", color=RED, ms=5, mew=1.2)
        tag = ("model ✓" if r["m"]["correct"] else "model ✗") + " · " + ("baseline ✓" if r["b"] else "baseline ✗")
        ax.set_title(f"{str(data.obj_names[k]).replace('_', ' ')[:12]}\n{tag}", fontsize=7, color=INK)
        axq.set_title("\nQ map", fontsize=7, color=MUTED)
        pts = np.concatenate([rect_to_corners(gt) for gt in data.rects(k)] + [rect_to_corners(r["g"])])
        for a_ in (ax, axq):
            zoom(a_, pts, 45); clean(a_)
    handles = [plt.Line2D([], [], color=RED, lw=2, label="model grasp"), plt.Line2D([], [], color=MUTED, lw=2, label="baseline grasp"),
               plt.Line2D([], [], color=YELLOW, lw=1, label="labelled grasps")]
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.3)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    save(fig, "model_examples")
    f = chosen[-1]
    return {"FailName": str(data.obj_names[f["k"]]).replace("_", " ").lower(), "FailIoU": f"{f['m']['best_iou']:.2f}",
            "FailAngle": f"{f['m']['min_angle_diff_deg']:.0f}",
            "FailIoUAngleOk": f"{f['m']['best_iou_angle_ok']:.2f}"}


# ------------------------------------------------------------------ tables
def data_table(split_stats):
    o, i = split_stats["objectwise"], split_stats["imagewise"]
    subs = ("train", "val", "test")
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Object-wise} & \multicolumn{3}{c}{Image-wise} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}", r" & train & val & test & train & val & test \\", r"\midrule"]
    for key, lab in (("images", "Images"), ("objects", "Objects"), ("rects", "Grasp rects")):
        lines.append(lab + " & " + " & ".join(str(s[x][key]) for s in (o, i) for x in subs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def results_table(bm, ev_o, ev_i, timing):
    f = lambda s: f"{100 * s['accuracy']:.1f} \\scriptsize({100 * s['ci95'][0]:.0f}--{100 * s['ci95'][1]:.0f})"  # noqa: E731
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Test acc.\ (\%) & Object-w. & Image-w. & ms \\", r"\midrule",
             f"PCA baseline & {f(bm['objectwise/test'])} & {f(bm['imagewise/test'])} & "
             f"{bm['objectwise/test']['ms_per_image_median']:.1f}$^c$ \\\\",
             f"Our CNN & \\textbf{{{f(ev_o['test'])}}} & \\textbf{{{f(ev_i['test'])}}} & {timing['ms_median']:.1f}$^g$ \\\\",
             r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ numbers
def numbers_tex(d):
    lines = ["% Generated by scripts/make_progress_report_assets.py -- do not edit."]
    for k, v in d.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    return "\n".join(lines) + "\n"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}\\%"


def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(repo_path("configs/cornell_rgbd.yaml"))
    cfg_i = load_config(repo_path("configs/cornell_rgbd_imagewise.yaml"))
    dcfg = load_config(repo_path(cfg["data_config"]))
    bcfg = load_config(repo_path("configs/baseline.yaml"))
    ecfg, lab = cfg["eval"], dcfg["label_maps"]
    data = CachedGraspData(repo_path(dcfg["paths"]["cache"]))

    pre = json.load(open(repo_path("runs/data_stats/preprocess_log.json")))
    ss = json.load(open(repo_path("runs/data_stats/split_stats.json")))
    bm = json.load(open(repo_path(f"runs/{bcfg['name']}/metrics.json")))
    mm = json.load(open(repo_path(f"runs/{cfg['name']}/metrics.json")))
    ev_o = json.load(open(repo_path(f"runs/{cfg['name']}/eval.json")))
    ev_i = json.load(open(repo_path(f"runs/{cfg_i['name']}/eval.json")))
    fs = json.load(open(repo_path(f"runs/{cfg['name']}/figure_stats.json")))
    log = list(csv.DictReader(open(repo_path(f"runs/{cfg['name']}/log.csv"))))
    vacc = np.array([float(r["val_acc"]) for r in log])
    ep_time = np.median([float(r["time_s"]) for r in log])
    with open(repo_path(f"runs/{bcfg['name']}/per_image_objectwise_test.csv")) as f:
        brows = list(csv.DictReader(f))

    # figures
    model, _ = load_model(cfg, device)
    val_ds = GraspDataset(data, repo_path(cfg["data"]["split"]), "val", cfg["data"]["channels"], label_cfg=lab)
    test_ds = GraspDataset(data, repo_path(cfg["data"]["split"]), "test", cfg["data"]["channels"], label_cfg=lab)
    val_recs = model_outputs(model, val_ds, device, ecfg, lab)
    good = [r for r in val_recs if r["m"]["correct"] and r["m"]["best_iou_angle_ok"] > 0.5]
    teaser(data, max(good, key=lambda r: (data.depth[r["k"]] > 0.01).sum()), ecfg)  # a validation image, not test
    data_example(data, dcfg, 100)
    baseline_examples(data, bcfg["baseline"], brows)
    fail_info = model_examples(data, bcfg["baseline"], model_outputs(model, test_ds, device, ecfg, lab), ecfg)
    # figures redrawn at their printed size so fonts stay >= 6 pt
    with plt.rc_context({"font.size": 6.5}):
        mbf.diagram(data, bcfg["baseline"], 462, FIG, figsize=(5.4, 1.4), name="baseline_diagram", title_fs=6.5,
                    titles=["① table level =\nborder median", "② mask: > 1 cm\nabove table", "③ largest comp.\nnot on border",
                            "④ centroid +\nPCA axes", "⑤ grasp across axis\nw = extent+10 px"])
        mmf.learning_curves(repo_path(f"runs/{cfg['name']}"), FIG, bm["objectwise/val"]["accuracy"], mm["best_epoch"],
                            figsize=(3.15, 1.15), name="learning_curves", short=True)
        mmf.architecture(FIG, mm["params"], figsize=(4.9, 1.45), fs=0.88, header=False, name="architecture")

    # tables
    open(os.path.join(TAB, "data_stats.tex"), "w").write(data_table(ss))
    open(os.path.join(TAB, "results.tex"), "w").write(results_table(bm, ev_o, ev_i, ev_o["timing"]))
    shutil.copy(repo_path(f"runs/{cfg['name']}/layer_table.tex"), os.path.join(TAB, "layers.tex"))

    o = ss["objectwise"]
    n = {
        "NImages": pre["images_kept"], "NObjects": pre["objects_kept"], "NRectsRaw": f"{pre['rects']['raw']:,}",
        "NRects": f"{pre['rects']['kept']:,}", "NRectsDegenerate": pre["rects"]["degenerate"],
        "NImagesDropped": pre["images_raw"] - pre["images_kept"],
        "MissingDepthCrop": pct(pre["missing_depth_frac_kept"]["mean"]),
        "MissingDepthCropMax": pct(pre["missing_depth_frac_kept"]["max"], 0),
        "TableInlierMedian": pct(pre["table_plane_inlier_frac"]["median"], 0),
        "TrainImgs": o["train"]["images"], "ValImgs": o["val"]["images"], "TestImgs": o["test"]["images"],
        "TrainObjs": o["train"]["objects"], "ValObjs": o["val"]["objects"], "TestObjs": o["test"]["objects"],
        "BaseObjVal": pct(bm["objectwise/val"]["accuracy"]), "BaseObjTest": pct(bm["objectwise/test"]["accuracy"]),
        "BaseImgTest": pct(bm["imagewise/test"]["accuracy"]),
        "BaseFailAngle": bm["objectwise/test"]["failures_angle"], "BaseFailIoU": bm["objectwise/test"]["failures_iou_only"],
        "BaseMs": f"{bm['objectwise/test']['ms_per_image_median']:.1f}",
        "Params": f"{mm['params'] / 1e6:.2f}\\,M", "ParamsExact": f"{mm['params']:,}",
        "BestEpoch": mm["best_epoch"], "Epochs": mm["epochs"], "EpochSec": f"{ep_time:.1f}",
        "TrainMin": f"{mm['train_time_s'] / 60:.0f}",
        "ModelObjVal": pct(ev_o["val"]["accuracy"]), "ModelObjTest": pct(ev_o["test"]["accuracy"]),
        "ModelObjTestLo": pct(ev_o["test"]["ci95"][0], 0), "ModelObjTestHi": pct(ev_o["test"]["ci95"][1], 0),
        "ModelImgVal": pct(ev_i["val"]["accuracy"]), "ModelImgTest": pct(ev_i["test"]["accuracy"]),
        "ModelFailAngle": ev_o["test"]["failures_angle"], "ModelFailIoU": ev_o["test"]["failures_iou_only"],
        "ModelMs": f"{ev_o['timing']['ms_median']:.1f}",
        "ValAccMinLate": pct(vacc[20:].min(), 0), "ValAccMaxLate": pct(vacc[20:].max(), 0),
        "ValAccMinAll": pct(vacc.min(), 0),
        "FixesBaseline": fs["objectwise_test_vs_baseline"]["model_fixes_baseline"],
        "BreaksBaseline": fs["objectwise_test_vs_baseline"]["model_breaks_baseline"],
    }
    n.update(fail_info)
    open(os.path.join(OUT, "numbers.tex"), "w").write(numbers_tex(n))
    print(json.dumps(n, indent=1))


if __name__ == "__main__":
    main()
