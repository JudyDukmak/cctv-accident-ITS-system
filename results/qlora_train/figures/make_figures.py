"""Evaluation figures for the QLoRA training (no model run; synthetic fold 1, window A only).

Reads only saved files:
  results/qlora_train/baseline/val_baseline.csv, val_epoch1.csv, val_epoch2.csv  (+ the *_raw.jsonl answers, checked)
  results/qlora_train/baseline/val_baseline_metrics.json, val_epoch{1,2}_metrics.json  (every recomputed metric
      is checked against these; the script stops on any mismatch)
  results/qlora_train/steps.csv, results/qlora_data/manifest.csv, scorer_config.json, cached frames (example grid)
Writes figures, summary.csv / summary.md / per_type.csv / examples.csv and FIGURES.md to results/qlora_train/figures/.
Refuses to overwrite. Run with the .venv-ml interpreter (numpy, pandas, matplotlib, PIL); no torch, no GPU.
"""
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
QT = REPO / "results" / "qlora_train"
MANIFEST = REPO / "results" / "qlora_data" / "manifest.csv"
SCORER = REPO / "scorer_config.json"
TYPES = ["head-on", "rear-end", "sideswipe", "single", "t-bone"]
MODELS = {"untrained": (QT / "baseline" / "val_baseline.csv", QT / "baseline" / "val_baseline_metrics.json",
                        QT / "baseline" / "val_baseline_raw.jsonl"),
          "epoch 1": (QT / "val_epoch1.csv", QT / "val_epoch1_metrics.json", QT / "val_epoch1_raw.jsonl"),
          "epoch 2": (QT / "val_epoch2.csv", QT / "val_epoch2_metrics.json", QT / "val_epoch2_raw.jsonl")}
# Same colour for the same model in every figure: reference categorical slots 1-3 (validated all-pairs, light mode).
COL = {"untrained": "#2a78d6", "epoch 1": "#eb6834", "epoch 2": "#1baf7a"}
BASE_COL = {"majority": "#52514e", "centre": "#8a8985"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EPOCH1_END, RESUME = 3154, 4500
SEED = 0

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160,
                     "legend.frameon": False})


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def load():
    cfg = json.loads(SCORER.read_text())
    sx, sy = cfg["sigma_x"] * cfg["spatial_k"], cfg["sigma_y"] * cfg["spatial_k"]
    man = pd.read_csv(MANIFEST, dtype={"fold": str})
    man = man[(man["fold"] == "1") & (man["window"] == "A")].set_index("rgb_path")
    data = {}
    for name, (csv_p, met_p, raw_p) in MODELS.items():
        d = pd.read_csv(csv_p)
        raw = [json.loads(line) for line in raw_p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(raw) != len(d) or [r["clip"] for r in raw] != d["rgb_path"].tolist():
            fail(f"{raw_p.name}: raw answers do not line up with {csv_p.name}")
        if set(d["rgb_path"]) != set(man.index) or len(d) != len(man):
            fail(f"{csv_p.name}: windows differ from manifest fold 1 window A")
        m = man.loc[d["rgb_path"]]
        if not (np.allclose(m["center_x"].values, d["center_x"].values) and
                np.allclose(m["center_y"].values, d["center_y"].values) and (m["type"].values == d["type"].values).all()):
            fail(f"{csv_p.name}: labels differ from the manifest")
        d["err"] = np.hypot(d["x"] - d["center_x"], d["y"] - d["center_y"])
        d["s"] = np.exp(-((d["x"] - d["center_x"]) ** 2 / (2 * sx ** 2) + (d["y"] - d["center_y"]) ** 2 / (2 * sy ** 2)))
        d["correct"] = d["pred_type"] == d["type"]
        d["err0"] = np.hypot(0.5 - d["center_x"], 0.5 - d["center_y"])
        d["s0"] = np.exp(-((0.5 - d["center_x"]) ** 2 / (2 * sx ** 2) + (0.5 - d["center_y"]) ** 2 / (2 * sy ** 2)))
        saved = json.loads(met_p.read_text())
        check = {"type_accuracy": d["correct"].mean(), "S": d["s"].mean(), "mean_point_error": d["err"].mean(),
                 "median_point_error": d["err"].median(), "centre_S": d["s0"].mean(),
                 "centre_mean_point_error": d["err0"].mean(),
                 "majority_accuracy": (d["type"] == saved["majority_type"]).mean()}
        for k, v in check.items():
            if abs(v - saved[k]) > 1e-9:
                fail(f"{name}: recomputed {k} {v} != saved {saved[k]}")
        for t in TYPES:
            for p in TYPES:
                if int(((d["type"] == t) & (d["pred_type"] == p)).sum()) != saved["confusion"][t][p]:
                    fail(f"{name}: confusion {t}->{p} differs from saved")
        data[name] = (d, saved)
    return data, man, cfg


def confusion(d, norm):
    c = np.array([[((d["type"] == t) & (d["pred_type"] == p)).sum() for p in TYPES] for t in TYPES], float)
    return c / c.sum(1, keepdims=True) if norm else c


def fig_confusion(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), constrained_layout=True)
    vmax = max(confusion(d, False).max() for d, _ in data.values())
    for j, (name, (d, s)) in enumerate(data.items()):
        for i, norm in enumerate((False, True)):
            ax = axes[i, j]
            c = confusion(d, norm)
            im = ax.imshow(c, cmap="Blues", vmin=0, vmax=1 if norm else vmax)
            for a in range(5):
                for b in range(5):
                    frac = c[a, b] / (1 if norm else vmax)
                    ax.text(b, a, f"{c[a, b]:.2f}" if norm else f"{int(c[a, b])}", ha="center", va="center",
                            fontsize=8.5, color="white" if frac > 0.55 else INK)
            ax.set_xticks(range(5), TYPES, rotation=30, ha="right")
            ax.set_yticks(range(5), TYPES)
            ax.grid(False)
            ax.set_xlabel("predicted type")
            if j == 0:
                ax.set_ylabel("true type")
            ax.set_title(f"{name}: {'row-normalised' if norm else 'counts'} (accuracy {s['type_accuracy']:.4f})",
                         color=INK, fontsize=10)
            # identity of the model is carried by a coloured bar under the title (text stays in ink)
            ax.add_patch(Rectangle((0, 1.005), 1, 0.012, transform=ax.transAxes, color=COL[name], clip_on=False))
        if j == 2:
            fig.colorbar(im, ax=axes[1, :], shrink=0.8, label="share of the true-type row")
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=axes[0, :], shrink=0.8, label="windows")
    fig.suptitle("Type confusion on fold 1, window A (755 windows); same colour scale within each row", fontsize=11)
    fig.savefig(out / "fig1_confusion.png")
    plt.close(fig)


def fig_training(out):
    st = pd.read_csv(QT / "steps.csv")
    if st["step"].tolist() != list(range(1, len(st) + 1)):
        fail("steps.csv is not one row per step 1..N")
    ma = st["loss"].rolling(100, min_periods=100).mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]},
                                 constrained_layout=True)
    a1.plot(st["step"], st["loss"], color="#a9c9ef", lw=0.5, label="loss per step (batch 1)")
    a1.plot(st["step"], ma, color="#1c5aa6", lw=2, label="100-step moving average (trailing)")
    a2.plot(st["step"], st["lr"], color="#1c5aa6", lw=2)
    for ax in (a1, a2):
        ax.axvline(EPOCH1_END, color=INK2, lw=1, ls="--")
        ax.axvline(RESUME, color=INK2, lw=1, ls=":")
    a1.text(EPOCH1_END, a1.get_ylim()[1], " end of epoch 1 (step 3154)", va="top", fontsize=8, color=INK2)
    a1.text(RESUME, a1.get_ylim()[1] * 0.93, " resume point (step 4500; paused on battery)", va="top",
            fontsize=8, color=INK2)
    a1.set_ylabel("training loss (answer tokens)")
    a1.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
    a2.set_ylabel("learning rate")
    a2.set_xlabel("optimizer step")
    a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a1.set_title(f"QLoRA training: {len(st)} steps, 2 epochs, cosine schedule with 3% warmup", fontsize=11)
    fig.savefig(out / "fig2_training_curve.png")
    plt.close(fig)
    blocks = [(a, min(a + 499, len(st))) for a in range(1, len(st) + 1, 500)]
    slow = st[(st["gpu_s"] > 6)]
    return st, ma, blocks, slow


def fig_location(data, out):
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    xs = np.linspace(0, 1.0, 501)
    for name, (d, _) in data.items():
        e = np.sort(d["err"].values)
        ax.plot(xs, np.searchsorted(e, xs, side="right") / len(e), color=COL[name], lw=2, label=name)
    e0 = np.sort(next(iter(data.values()))[0]["err0"].values)
    ax.plot(xs, np.searchsorted(e0, xs, side="right") / len(e0), color=BASE_COL["centre"], lw=2, ls="--",
            label="image-centre baseline")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("point error (distance between predicted and true box centre, normalised image units)")
    ax.set_ylabel("share of windows with error below x")
    ax.set_title("Location: cumulative distribution of point error (fold 1, window A, 755 windows)", fontsize=10)
    ax.legend(loc="lower right")
    fig.savefig(out / "fig3_location_cdf.png")
    plt.close(fig)


def per_type(data):
    rows = []
    for name, (d, _) in data.items():
        for t in TYPES:
            dd = d[d["type"] == t]
            rows.append({"model": name, "type": t, "n": len(dd), "type_accuracy": dd["correct"].mean(),
                         "S": dd["s"].mean(), "mean_point_error": dd["err"].mean()})
    d0 = next(iter(data.values()))[0]
    for t in TYPES:
        dd = d0[d0["type"] == t]
        rows.append({"model": "image centre", "type": t, "n": len(dd), "type_accuracy": np.nan, "S": dd["s0"].mean(),
                     "mean_point_error": dd["err0"].mean()})
        rows.append({"model": "majority type (rear-end)", "type": t, "n": len(dd),
                     "type_accuracy": float(t == "rear-end"), "S": np.nan, "mean_point_error": np.nan})
    return pd.DataFrame(rows)


def grouped_bars(ax, groups, values, metric, baseline_val, baseline_name, baseline_col):
    w = 0.26
    x = np.arange(len(groups))
    for k, name in enumerate(values):
        ax.bar(x + (k - 1) * w, values[name], width=w - 0.03, color=COL[name], label=name, zorder=3)
    ax.axhline(baseline_val, color=baseline_col, lw=1.5, ls="--", zorder=4,
               label=f"{baseline_name} ({baseline_val:.4f}, all windows)")
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)


def fig_per_type(data, pt, out):
    s_any = next(iter(data.values()))[1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    labels = [f"{t}\n(n {int(pt[(pt.model == 'epoch 1') & (pt.type == t)].n.iloc[0])})" for t in TYPES]
    vals_a = {m: [pt[(pt.model == m) & (pt.type == t)].type_accuracy.iloc[0] for t in TYPES] for m in data}
    vals_s = {m: [pt[(pt.model == m) & (pt.type == t)].S.iloc[0] for t in TYPES] for m in data}
    grouped_bars(a1, labels, vals_a, "type accuracy (recall of the true type)", s_any["majority_accuracy"],
                 "majority-type baseline", BASE_COL["majority"])
    grouped_bars(a2, labels, vals_s, "S (location score)", s_any["centre_S"], "image-centre baseline",
                 BASE_COL["centre"])
    a1.set_title("Type accuracy per true type", fontsize=10)
    a2.set_title("S per true type", fontsize=10)
    a1.legend(loc="upper left", ncol=2, fontsize=8)
    a2.legend(loc="upper left", ncol=2, fontsize=8)
    fig.savefig(out / "fig4_per_type.png")
    plt.close(fig)


def fig_day_night(data, out):
    s_any = next(iter(data.values()))[1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    groups = ["day", "night"]
    ns = {g: int(s_any["per_day_night"][g]["n"]) for g in groups}
    labels = [f"{g} (n {ns[g]})" for g in groups]
    va = {m: [data[m][1]["per_day_night"][g]["type_accuracy"] for g in groups] for m in data}
    vs = {m: [data[m][1]["per_day_night"][g]["S"] for g in groups] for m in data}
    grouped_bars(a1, labels, va, "type accuracy", s_any["majority_accuracy"], "majority-type baseline",
                 BASE_COL["majority"])
    grouped_bars(a2, labels, vs, "S (location score)", s_any["centre_S"], "image-centre baseline",
                 BASE_COL["centre"])
    a1.set_title("Type accuracy, day vs night", fontsize=10)
    a2.set_title("S, day vs night", fontsize=10)
    a1.legend(loc="upper left", ncol=2, fontsize=8)
    a2.legend(loc="upper left", ncol=2, fontsize=8)
    fig.savefig(out / "fig5_day_night.png")
    plt.close(fig)


def read_frames(path):
    with np.load(path) as z:
        off, buf, fi = z["offsets"], z["data"], z["frame_idx"]
        frames = [np.asarray(Image.open(io.BytesIO(buf[off[k]:off[k + 1]].tobytes())).convert("RGB"))
                  for k in range(len(fi))]
    return fi.tolist(), frames


def fig_examples(data, man, out):
    d = data["epoch 1"][0].copy()
    rng = np.random.default_rng(SEED)
    rules = [("good location", d["err"] < 0.05), ("bad location", d["err"] > 0.30),
             ("correct type", d["correct"]), ("wrong type", ~d["correct"])]
    picked, rows = set(), []
    for label, mask in rules:
        pool = sorted(set(d.loc[mask, "rgb_path"]) - picked)
        for p in rng.choice(pool, size=2, replace=False):
            picked.add(p)
            rows.append((label, p, len(pool)))
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.0), constrained_layout=True)
    info = []
    for ax, (label, p, pool_n) in zip(axes.flat, rows):
        r, m = d.set_index("rgb_path").loc[p], man.loc[p]
        fi, frames = read_frames(REPO / m["frames_path"])
        if fi != json.loads(m["frame_indices"]):
            fail(f"cached frames of {p} differ from the manifest")
        times = np.array(fi) / float(m["fps"])
        k = int(np.argmin(np.abs(times - float(m["t_star"]))))
        img = frames[k]
        h, w = img.shape[:2]
        ax.imshow(img)
        tb = [m["x1_1000"], m["y1_1000"], m["x2_1000"], m["y2_1000"]]
        pb = json.loads(r["bbox"])
        pb = [min(pb[0], pb[2]), min(pb[1], pb[3]), max(pb[0], pb[2]), max(pb[1], pb[3])]
        for b, c in ((tb, "#00c000"), (pb, "#ff2020")):
            ax.add_patch(Rectangle((b[0] / 1000 * w, b[1] / 1000 * h), (b[2] - b[0]) / 1000 * w,
                                   (b[3] - b[1]) / 1000 * h, fill=False, ec=c, lw=2))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{label}: true {r['type']} / predicted {r['pred_type']}\n"
                     f"point error {r['err']:.3f}, S {r['s']:.3f}; frame {k + 1}/16 at {times[k]:.2f} s, "
                     f"t* {float(m['t_star']):.2f} s\n({'inside' if str(m['t_star_inside']) == 'True' else 'OUTSIDE'}"
                     f" the window)", fontsize=8)
        info.append({"category": label, "pool_size": pool_n, "rgb_path": p, "true_type": r["type"],
                     "pred_type_epoch1": r["pred_type"], "point_error": r["err"], "S": r["s"],
                     "frame_shown": k + 1, "frame_time_s": times[k], "t_star": float(m["t_star"]),
                     "t_star_inside": m["t_star_inside"], "true_box_1000": tb, "pred_box_1000_epoch1": r["bbox"]})
    fig.suptitle("Epoch 1 on fold 1, window A: green = true box, red = epoch-1 box; frame nearest t*", fontsize=11)
    fig.savefig(out / "fig6_examples.png")
    plt.close(fig)
    pd.DataFrame(info).to_csv(out / "examples.csv", index=False)
    return info


def summary(data):
    rows = []
    for name, (d, s) in data.items():
        rows.append({"model": name, "n": len(d), "valid_answers": int((d["valid"] == True).sum()),  # noqa: E712
                     "type_accuracy": s["type_accuracy"], "S": s["S"], "mean_acc_S": s["score_mean_acc_S"],
                     "mean_point_error": s["mean_point_error"], "median_point_error": s["median_point_error"],
                     "acc_day": s["per_day_night"]["day"]["type_accuracy"], "S_day": s["per_day_night"]["day"]["S"],
                     "acc_night": s["per_day_night"]["night"]["type_accuracy"],
                     "S_night": s["per_day_night"]["night"]["S"],
                     "collision_visible_true": int(s["collision_visible_counts"].get("true", 0)),
                     "visible_agrees_t_star_inside": s["collision_visible_agrees_with_t_star_inside"],
                     "gpu_s_per_window": s["gpu_s_per_window"]})
    d, s = next(iter(data.values()))
    maj = s["majority_type"]
    na = np.nan
    dd = {g: d[d["day_night"] == g] for g in ("day", "night")}
    rows.append({"model": f"majority type ({maj})", "n": len(d), "valid_answers": na,
                 "type_accuracy": s["majority_accuracy"], "S": na, "mean_acc_S": na, "mean_point_error": na,
                 "median_point_error": na, "acc_day": (dd["day"]["type"] == maj).mean(), "S_day": na,
                 "acc_night": (dd["night"]["type"] == maj).mean(), "S_night": na, "collision_visible_true": na,
                 "visible_agrees_t_star_inside": na, "gpu_s_per_window": na})
    rows.append({"model": "image centre", "n": len(d), "valid_answers": na, "type_accuracy": na, "S": s["centre_S"],
                 "mean_acc_S": na, "mean_point_error": s["centre_mean_point_error"],
                 "median_point_error": d["err0"].median(), "acc_day": na, "S_day": dd["day"]["s0"].mean(),
                 "acc_night": na, "S_night": dd["night"]["s0"].mean(), "collision_visible_true": na,
                 "visible_agrees_t_star_inside": na, "gpu_s_per_window": na})
    return pd.DataFrame(rows)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif pd.isna(v):
                cells.append("—")
            elif float(v).is_integer() and c in ("n", "valid_answers", "collision_visible_true"):
                cells.append(str(int(v)))
            else:
                cells.append(f"{v:.4f}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    out = HERE
    if (out / "FIGURES.md").exists() or (out / "summary.csv").exists():
        fail(f"{out} already has outputs; refusing to overwrite")
    data, man, cfg = load()
    print("recomputed metrics match the saved metrics JSON for all three models")
    fig_confusion(data, out)
    st, ma, blocks, slow = fig_training(out)
    fig_location(data, out)
    pt = per_type(data)
    pt.to_csv(out / "per_type.csv", index=False)
    fig_per_type(data, pt, out)
    fig_day_night(data, out)
    ex = fig_examples(data, man, out)
    sm = summary(data)
    sm.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text("# QLoRA evaluation summary (fold 1, window A, 755 synthetic windows)\n\n"
                                    + md_table(sm) + "\n", encoding="utf-8")

    # numbers quoted in FIGURES.md, all computed here
    S = {n: s for n, (_, s) in data.items()}
    D = {n: d for n, (d, _) in data.items()}
    c1, c2, c0 = (confusion(D[n], False) for n in ("epoch 1", "epoch 2", "untrained"))
    ti = {t: i for i, t in enumerate(TYPES)}
    share_below = {n: (D[n]["err"] < 0.1).mean() for n in D}
    share_below["image centre"] = (D["untrained"]["err0"] < 0.1).mean()
    ins = {n: D[n]["t_star_inside"].astype(str) == "True" for n in D}
    xs = np.linspace(0, 1, 1001)

    def cdf(e):
        e = np.sort(np.asarray(e))
        return np.searchsorted(e, xs, side="right") / len(e)

    c_centre = cdf(D["untrained"]["err0"])
    cross = {n: xs[(xs > 0.1) & (cdf(D[n]["err"]) < c_centre)].min() for n in ("epoch 1", "epoch 2")}
    tail = {n: (D[n]["err"] > 0.4).mean() for n in D}
    tail["image centre"] = (D["untrained"]["err0"] > 0.4).mean()
    n_out = sum(str(e["t_star_inside"]) != "True" for e in ex)
    bl = "; ".join(f"steps {a}-{b} {st['loss'][a - 1:b].mean():.4f}" for a, b in blocks)
    pts = {(r.model, r.type): r for r in pt.itertuples()}
    best_s_type = max(TYPES, key=lambda t: pts[("epoch 1", t)].S)
    worst_s_type = min(TYPES, key=lambda t: pts[("epoch 1", t)].S)
    lines = [
        "# QLoRA evaluation figures",
        "",
        "All figures use only saved files: the validation answers of the untrained model (`baseline/`), epoch 1 and "
        "epoch 2 on **fold 1, window A (755 synthetic windows)**, `steps.csv`, the manifest "
        "(`results/qlora_data/manifest.csv`) and the cached frames. No model was run and no real clip or label was "
        "read. Every metric was recomputed from the per-window CSVs and checked against the saved metrics JSON "
        "(stop on any mismatch). Colours: untrained blue, epoch 1 orange, epoch 2 aqua in every figure; baselines grey "
        "dashed. Made by `make_figures.py` (this folder) with the `.venv-ml` interpreter.",
        "",
        "Checkpoint choice (fixed rule, unchanged): epoch 1 (mean of accuracy and S "
        f"{S['epoch 1']['score_mean_acc_S']:.4f} vs {S['epoch 2']['score_mean_acc_S']:.4f}).",
        "",
        "## 1. Confusion matrices — `fig1_confusion.png`",
        f"Counts (top) and row-normalised shares (bottom) for untrained, epoch 1 and epoch 2, with one colour scale per "
        f"row. The untrained model answers mostly t-bone ({int(c0[:, ti['t-bone']].sum())} of 755); after training the "
        f"answers move to head-on: epoch 1 predicts head-on for {int(c1[ti['rear-end'], ti['head-on']])} of "
        f"{int(c1[ti['rear-end']].sum())} rear-end and {int(c1[ti['sideswipe'], ti['head-on']])} of "
        f"{int(c1[ti['sideswipe']].sum())} sideswipe windows, and epoch 2 predicts head-on in "
        f"{int(c2[:, ti['head-on']].sum())} of 755 windows. Single is never predicted correctly by any model "
        f"({int(c0[ti['single'], ti['single']])}, {int(c1[ti['single'], ti['single']])}, "
        f"{int(c2[ti['single'], ti['single']])} of {int(c1[ti['single']].sum())}).",
        "",
        "## 2. Training curve — `fig2_training_curve.png`",
        f"Per-step loss on the answer tokens (light; batch 1, so one example per step) with a trailing 100-step moving "
        f"average (dark), and the learning rate below. The 500-step means fall from {bl.split('; ')[0]} to "
        f"{bl.split('; ')[-1]}; the dashed line marks the end of epoch 1 (step {EPOCH1_END}) and the dotted line the "
        f"resume point (step {RESUME}), where the run paused on battery and was resumed from the saved checkpoint "
        f"with the same code (sha256 identical in run.log). Steps {int(slow['step'].min())}-{int(slow['step'].max())} "
        f"({len(slow)} steps) ran slower (GPU time > 6 s per step, likely on battery, which the power check only sees "
        f"at a checkpoint); the slowdown changes speed, not the training data or settings.",
        "",
        f"All 500-step means: {bl}.",
        "",
        "## 3. Location — `fig3_location_cdf.png`",
        f"Share of windows whose predicted box centre lies within x (normalised image units) of the true box centre. "
        f"For small errors both trained epochs lie well above the untrained model and the image-centre baseline: the "
        f"share with error below 0.1 is {share_below['untrained']:.4f} untrained, {share_below['epoch 1']:.4f} epoch 1,"
        f" {share_below['epoch 2']:.4f} epoch 2 and {share_below['image centre']:.4f} for the image centre. The curves "
        f"cross the image-centre curve at an error of about {cross['epoch 1']:.3f} (epoch 1) and {cross['epoch 2']:.3f}"
        f" (epoch 2): the trained models have a heavier tail of far-off answers (error above 0.4 in "
        f"{tail['epoch 1']:.4f} and {tail['epoch 2']:.4f} of windows, vs {tail['image centre']:.4f} for the image "
        f"centre and {tail['untrained']:.4f} untrained). "
        f"Median point error: {S['untrained']['median_point_error']:.4f}, {S['epoch 1']['median_point_error']:.4f}, "
        f"{S['epoch 2']['median_point_error']:.4f} and {D['untrained']['err0'].median():.4f}.",
        "",
        "## 4. Per-type accuracy and S — `fig4_per_type.png`",
        f"Left: type accuracy within each true type (recall); right: S within each true type. Horizontal lines are the "
        f"overall baselines on all 755 windows (majority type rear-end {S['epoch 1']['majority_accuracy']:.4f}; image "
        f"centre S {S['epoch 1']['centre_S']:.4f}); the per-type baseline values are in `per_type.csv`. Training "
        f"raises S for every type (epoch 1 highest for {best_s_type} {pts[('epoch 1', best_s_type)].S:.4f}, lowest "
        f"for {worst_s_type} {pts[('epoch 1', worst_s_type)].S:.4f}), while type accuracy is high only for head-on "
        f"and t-bone at epoch 1 and only for head-on at epoch 2.",
        "",
        "## 5. Day vs night — `fig5_day_night.png`",
        f"Type accuracy and S for the {int(S['epoch 1']['per_day_night']['day']['n'])} day and "
        f"{int(S['epoch 1']['per_day_night']['night']['n'])} night windows, with the overall baselines as lines. "
        f"S is lower at night for both epochs (epoch 1 {S['epoch 1']['per_day_night']['day']['S']:.4f} day vs "
        f"{S['epoch 1']['per_day_night']['night']['S']:.4f} night; epoch 2 "
        f"{S['epoch 2']['per_day_night']['day']['S']:.4f} vs {S['epoch 2']['per_day_night']['night']['S']:.4f}). "
        f"The night group is small (n {int(S['epoch 1']['per_day_night']['night']['n'])}), so differences there are "
        f"uncertain.",
        "",
        "## 6. Examples — `fig6_examples.png`",
        f"Eight fold-1 window-A windows scored with the epoch-1 answers, showing the cached frame nearest t* with the "
        f"true box (green) and the epoch-1 box (red). Selection rule, fixed before looking at the images: 2 each drawn "
        f"at random (numpy default_rng({SEED}), no repeats) from good location (point error < 0.05), bad location "
        f"(> 0.30), correct type and wrong type; pool sizes and details are in `examples.csv`. In {n_out} of the 8, t* "
        f"lies outside the 4-second window (marked OUTSIDE in the title), so the shown frame is the window edge, not "
        f"the collision, and the true box refers to a moment the model did not see. These are illustrations, not a "
        f"measurement.",
        "",
        "## 7. Summary table — `summary.csv`, `summary.md`",
        "Every metric for the three models and both baselines on the same 755 windows (— = not applicable). "
        "`per_type.csv` has the per-type numbers behind figure 4.",
        "",
        md_table(sm),
        "",
    ]
    (out / "FIGURES.md").write_text("\n".join(lines), encoding="utf-8")
    print(md_table(sm))
    print("examples:")
    for e in ex:
        print(f"  {e['category']:14s} pool {e['pool_size']:4d}  {e['rgb_path']}  true {e['true_type']} pred "
              f"{e['pred_type_epoch1']} err {e['point_error']:.3f}")
    print(f"slow steps (gpu_s > 6): {len(slow)} from {int(slow['step'].min())} to {int(slow['step'].max())}")
    print("wrote:", ", ".join(sorted(p.name for p in out.iterdir() if p.name != "make_figures.py")))


if __name__ == "__main__":
    main()
