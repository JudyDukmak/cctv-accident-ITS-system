"""Inside vs outside analysis of the QLoRA validation (EXPLORATORY: decided after seeing figure 6).

Splits the 755 fold-1 window-A windows by whether t* lies inside window A (first frame time <= t* <= last frame time,
the manifest's t_star_inside). No model run. Reads only:
  results/qlora_train/baseline/val_baseline.csv, val_epoch1.csv, val_epoch2.csv   (per-window answers)
  results/qlora_data/manifest.csv   (synthetic labels; window A centre_s = classifier t_hat, C_cctv seed 0 out-of-fold)
  scorer_config.json   (sigma_x, sigma_y, spatial_k for S)
Writes CSVs, two PNGs and report.txt to results/qlora_train/inside_outside/. Refuses to overwrite.
Run with the .venv-ml interpreter (numpy, pandas, matplotlib); no torch, no GPU.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
QT = REPO / "results" / "qlora_train"
MANIFEST = REPO / "results" / "qlora_data" / "manifest.csv"
TYPES = ["head-on", "rear-end", "sideswipe", "single", "t-bone"]
MODELS = {"untrained": QT / "baseline" / "val_baseline.csv", "epoch 1": QT / "val_epoch1.csv",
          "epoch 2": QT / "val_epoch2.csv"}
GROUPS = ("inside", "outside", "all")
COL = {"inside": "#2a78d6", "outside": "#eb6834"}  # reference categorical slots 1-2
FAR = 0.4
EARLIER_SHARE = 0.7558  # results/qlora_data/report.txt: window A t* inside 1671/2211 = 0.7558

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 160, "legend.frameon": False})


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def f4(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def table_text(df, int_cols=("n",)):
    cols = list(df.columns)
    rows = [[str(c) for c in cols]]
    for _, r in df.iterrows():
        rows.append([str(r[c]) if isinstance(r[c], str) else (str(int(r[c])) if c in int_cols else f4(float(r[c])))
                     for c in cols])
    w = [max(len(x[i]) for x in rows) for i in range(len(cols))]
    return "\n".join("  " + "  ".join(x[i].rjust(w[i]) if i else x[i].ljust(w[i]) for i in range(len(cols)))
                     for x in rows)


def main():
    if any(p.name not in (Path(__file__).name, "run_log.txt") for p in HERE.iterdir()):
        fail(f"{HERE} already has outputs; refusing to overwrite")
    cfg = json.loads((REPO / "scorer_config.json").read_text())
    sx, sy = cfg["sigma_x"] * cfg["spatial_k"], cfg["sigma_y"] * cfg["spatial_k"]
    man = pd.read_csv(MANIFEST, dtype={"fold": str})
    A = man[man["window"] == "A"].copy()
    if not (A["centre_source"] == "t_hat C_cctv seed 0 out-of-fold").all():
        fail("window A centre is not the classifier t_hat for every row")
    A["inside"] = A["t_star_inside"].astype(str) == "True"
    A["abs_that_err"] = (A["centre_s"] - A["t_star"]).abs()
    m1 = A[A["fold"] == "1"].set_index("rgb_path")

    def s_err(x, y, gx, gy):
        return (np.exp(-((x - gx) ** 2 / (2 * sx ** 2) + (y - gy) ** 2 / (2 * sy ** 2))), np.hypot(x - gx, y - gy))

    data = {}
    for name, p in MODELS.items():
        d = pd.read_csv(p)
        if len(d) != 755 or set(d["rgb_path"]) != set(m1.index):
            fail(f"{p.name}: windows differ from manifest fold 1 window A")
        mm = m1.loc[d["rgb_path"]]
        if not ((mm["inside"].values == (d["t_star_inside"].astype(str) == "True").values).all()
                and np.allclose(mm["center_x"].values, d["center_x"].values) and (mm["type"].values == d["type"].values).all()):
            fail(f"{p.name}: labels or t_star_inside differ from the manifest")
        d["group"] = np.where(mm["inside"].values, "inside", "outside")
        d["s"], d["err"] = s_err(d["x"], d["y"], d["center_x"], d["center_y"])
        d["s0"], d["err0"] = s_err(0.5, 0.5, d["center_x"], d["center_y"])
        d["correct"] = d["pred_type"] == d["type"]
        d["night"] = mm["day_night"].values == "night"
        d["abs_that_err"] = mm["abs_that_err"].values
        data[name] = d
    base = data["untrained"]

    def grp(d, g):
        return d if g == "all" else d[d["group"] == g]

    # 1. model table
    t1 = pd.DataFrame([{"model": n, "group": g, "n": len(grp(d, g)), "type_accuracy": grp(d, g)["correct"].mean(),
                        "S": grp(d, g)["s"].mean(), "median_point_error": grp(d, g)["err"].median(),
                        "share_error_above_0.4": (grp(d, g)["err"] > FAR).mean()}
                       for n, d in data.items() for g in GROUPS])
    # sanity: "all" rows equal the saved metrics JSON
    for n, f in (("untrained", QT / "baseline" / "val_baseline_metrics.json"), ("epoch 1", QT / "val_epoch1_metrics.json"),
                 ("epoch 2", QT / "val_epoch2_metrics.json")):
        sv = json.loads(f.read_text())
        r = t1[(t1.model == n) & (t1.group == "all")].iloc[0]
        if abs(r.type_accuracy - sv["type_accuracy"]) > 1e-9 or abs(r.S - sv["S"]) > 1e-9:
            fail(f"{n}: recomputed overall metrics differ from {f.name}")
    t1.to_csv(HERE / "1_models_by_group.csv", index=False)

    # 2. baselines within each group
    rows = []
    for g in GROUPS:
        b = grp(base, g)
        maj = b["type"].value_counts().idxmax()
        rows.append({"group": g, "n": len(b), "group_majority_type": maj,
                     "group_majority_accuracy": (b["type"] == maj).mean(),
                     "rear_end_accuracy": (b["type"] == "rear-end").mean(), "centre_S": b["s0"].mean(),
                     "centre_median_point_error": b["err0"].median(),
                     "centre_share_error_above_0.4": (b["err0"] > FAR).mean()})
    t2 = pd.DataFrame(rows)
    t2.to_csv(HERE / "2_baselines_by_group.csv", index=False)

    # 3. row-normalised confusion, inside group only
    conf = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, n in zip(axes, ("epoch 1", "epoch 2")):
        d = grp(data[n], "inside")
        c = pd.DataFrame([[((d["type"] == t) & (d["pred_type"] == p)).sum() for p in TYPES] for t in TYPES],
                         index=TYPES, columns=TYPES)
        nrm = c.div(c.sum(axis=1), axis=0)
        conf[n] = (c, nrm)
        out = nrm.copy()
        out.insert(0, "n_true", c.sum(axis=1))
        out.to_csv(HERE / f"3_confusion_inside_{n.replace(' ', '')}_rownorm.csv", index_label="true\\predicted")
        ax.imshow(nrm.values, cmap="Blues", vmin=0, vmax=1)
        for a in range(5):
            for b_ in range(5):
                v = nrm.values[a, b_]
                ax.text(b_, a, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.55 else "#0b0b0b")
        ax.set_xticks(range(5), TYPES, rotation=30, ha="right")
        ax.set_yticks(range(5), [f"{t} (n {int(c.loc[t].sum())})" for t in TYPES])
        ax.grid(False)
        ax.set_xlabel("predicted type")
        ax.set_ylabel("true type")
        ax.set_title(f"{n}, t* inside window A (n {len(d)}): row-normalised", fontsize=10)
    fig.savefig(HERE / "3_confusion_inside.png")
    plt.close(fig)

    # 4. how the groups differ
    rows = []
    for g in GROUPS:
        b = grp(base, g)
        e = b["abs_that_err"]
        r = {"group": g, "n": len(b), "share_night": b["night"].mean()}
        r.update({f"share_{t}": (b["type"] == t).mean() for t in TYPES})
        r.update({"abs_that_minus_tstar_mean": e.mean(), "p05": e.quantile(0.05), "p25": e.quantile(0.25),
                  "median": e.median(), "p75": e.quantile(0.75), "p95": e.quantile(0.95), "max": e.max(),
                  "share_above_2s": (e > 2).mean()})
        rows.append(r)
    t4 = pd.DataFrame(rows)
    t4.to_csv(HERE / "4_group_differences.csv", index=False)
    # extra (stratified, for the harder-clip question): accuracy and S per true type within each group
    rows = []
    for n, d in data.items():
        for t in TYPES:
            for g in ("inside", "outside"):
                dd = grp(d, g)
                dd = dd[dd["type"] == t]
                rows.append({"model": n, "type": t, "group": g, "n": len(dd),
                             "type_accuracy": dd["correct"].mean() if len(dd) else np.nan,
                             "S": dd["s"].mean() if len(dd) else np.nan,
                             "centre_S": dd["s0"].mean() if len(dd) else np.nan})
    t4b = pd.DataFrame(rows)
    t4b.to_csv(HERE / "4b_per_type_by_group.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    bins = np.arange(0, np.ceil(base["abs_that_err"].max()) + 0.5, 0.5)
    for g in ("inside", "outside"):
        ax.hist(grp(base, g)["abs_that_err"], bins=bins, color=COL[g], alpha=0.75,
                label=f"t* {g} window A (n {len(grp(base, g))})")
    ax.set_xlabel("|classifier t_hat − t*| (s)")
    ax.set_ylabel("windows")
    ax.set_title("Fold 1: classifier timing error by group (0.5-s bins)", fontsize=10)
    ax.legend()
    fig.savefig(HERE / "4_abs_that_error_hist.png")
    plt.close(fig)

    # 5. share inside per fold (all window-A rows of the manifest)
    t5 = pd.DataFrame([{"fold": f, "n": int((A["fold"] == f).sum()) if f != "all" else len(A),
                        "n_inside": int(A.loc[A["fold"] == f, "inside"].sum()) if f != "all" else int(A["inside"].sum())}
                       for f in ("0", "1", "2", "all")])
    t5["share_inside"] = t5["n_inside"] / t5["n"]
    t5.to_csv(HERE / "5_share_inside_by_fold.csv", index=False)
    all_share = float(t5[t5.fold == "all"].share_inside.iloc[0])
    if abs(all_share - EARLIER_SHARE) > 5e-5:
        fail(f"overall share {all_share} does not reproduce the earlier {EARLIER_SHARE}")

    # 6. report
    T = {(r.model, r.group): r for r in t1.itertuples()}
    B = {r.group: r for r in t2.itertuples()}
    G = {r["group"]: r for r in t4.to_dict("records")}
    F = {r.fold: r for r in t5.itertuples()}
    gap = {n: (T[(n, "inside")].S - T[(n, "outside")].S, T[(n, "inside")].type_accuracy - T[(n, "outside")].type_accuracy)
           for n in data}
    gap0 = B["inside"].centre_S - B["outside"].centre_S
    lines = [
        "INSIDE vs OUTSIDE ANALYSIS — QLoRA validation, fold 1 window A (755 synthetic windows)",
        "EXPLORATORY: this split was decided after seeing figure 6 (results/qlora_train/figures/). It is not part of the evaluation",
        "fixed in CONFIG.txt before training, did not affect training or the checkpoint choice (epoch 1, unchanged), and no model was",
        "run. Inside = the first frame time <= t* <= the last frame time of window A (manifest t_star_inside). Window A is",
        "centred on the classifier's out-of-fold t_hat (C_cctv seed 0), so 'outside' means the classifier's t_hat was far",
        "from t*. Point error is in normalised image units; S uses scorer_config.json.",
        "",
        "1. Models by group (all-group rows reproduce the saved metrics JSON)",
        table_text(t1),
        "",
        "2. Baselines within each group (group majority = most common true type in that group; computed from labels)",
        table_text(t2),
        "",
        "3. Row-normalised confusion, t* inside only (rows true, columns predicted; n_true per row)",
    ]
    for n in ("epoch 1", "epoch 2"):
        c, nrm = conf[n]
        df = nrm.reset_index().rename(columns={"index": "true"})
        df.insert(1, "n", c.sum(axis=1).values)
        lines += [f"  {n}", table_text(df)]
    lines += [
        "",
        "4. How the groups differ (true labels; |t_hat - t*| in seconds)",
        table_text(t4[["group", "n", "share_night"] + [f"share_{t}" for t in TYPES]]),
        table_text(t4[["group", "abs_that_minus_tstar_mean", "p05", "p25", "median", "p75", "p95", "max",
                       "share_above_2s"]]),
        "",
        "4b. Extra (stratified): accuracy and S per true type within each group (small n in some cells)",
        table_text(t4b),
        "",
        "5. Share of windows with t* inside window A, per fold (manifest, all window-A rows)",
        table_text(t5, int_cols=("n", "n_inside")),
        f"  earlier number {EARLIER_SHARE} (results/qlora_data/report.txt) reproduced: all folds "
        f"{int(F['all'].n_inside)}/{int(F['all'].n)} = {F['all'].share_inside:.4f}",
        "",
        "6. Interpretation (exploratory; no causal claim)",
    ]
    lines += INTERPRETATION(T, B, G, F, gap, gap0, t4b)
    text = "\n".join(lines) + "\n"
    (HERE / "report.txt").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


def INTERPRETATION(T, B, G, F, gap, gap0, t4b):
    def pt(model, t, g, k):
        r = t4b[(t4b.model == model) & (t4b.type == t) & (t4b.group == g)].iloc[0]
        return r[k]

    # guards: every directional statement below is checked against the numbers
    if not all(pt(m, t, "inside", "S") > pt(m, t, "outside", "S") for m in ("epoch 1", "epoch 2") for t in TYPES):
        fail("S is not higher inside for every type; interpretation text must change")
    if not pt("epoch 1", "t-bone", "outside", "type_accuracy") >= pt("epoch 1", "t-bone", "inside", "type_accuracy"):
        fail("t-bone statement no longer holds")
    if not (B["outside"].centre_S > B["inside"].centre_S
            and abs(G["outside"]["share_night"] - G["inside"]["share_night"]) < 0.02):
        fail("centre-S or night-share statement no longer holds")
    if not (G["outside"]["share_rear-end"] > G["inside"]["share_rear-end"]
            and G["outside"]["share_sideswipe"] > G["inside"]["share_sideswipe"]
            and G["outside"]["share_head-on"] < G["inside"]["share_head-on"]):
        fail("type-mix statement no longer holds")
    return [
        f"  (1) Both trained epochs score much better when t* is inside window A (S {T[('epoch 1', 'inside')].S:.4f} vs "
        f"{T[('epoch 1', 'outside')].S:.4f} for epoch 1 and {T[('epoch 2', 'inside')].S:.4f} vs "
        f"{T[('epoch 2', 'outside')].S:.4f} for epoch 2; type accuracy {T[('epoch 1', 'inside')].type_accuracy:.4f} vs "
        f"{T[('epoch 1', 'outside')].type_accuracy:.4f} and {T[('epoch 2', 'inside')].type_accuracy:.4f} vs "
        f"{T[('epoch 2', 'outside')].type_accuracy:.4f}), while the untrained model's S gap is small "
        f"({gap['untrained'][0]:+.4f}).",
        f"  (2) The two groups are different clips, not the same clips with a different window: the outside group has "
        f"more rear-end ({G['outside']['share_rear-end']:.4f} vs {G['inside']['share_rear-end']:.4f}) and sideswipe "
        f"({G['outside']['share_sideswipe']:.4f} vs {G['inside']['share_sideswipe']:.4f}) and fewer head-on "
        f"({G['outside']['share_head-on']:.4f} vs {G['inside']['share_head-on']:.4f}), a similar night share "
        f"({G['outside']['share_night']:.4f} vs {G['inside']['share_night']:.4f}), and by construction a much larger "
        f"classifier timing error (median |t_hat - t*| {G['outside']['median']:.2f} s vs {G['inside']['median']:.2f} s), "
        f"so they are clips the classifier found hard to time and may be harder for other reasons too.",
        f"  (3) Much of the type-accuracy gap could come from this type mix rather than from the window: the trained "
        f"models almost never name rear-end or sideswipe in either group (epoch 1 rear-end accuracy "
        f"{pt('epoch 1', 'rear-end', 'inside', 'type_accuracy'):.4f} inside vs "
        f"{pt('epoch 1', 'rear-end', 'outside', 'type_accuracy'):.4f} outside), and for t-bone epoch 1 is not worse "
        f"outside ({pt('epoch 1', 't-bone', 'outside', 'type_accuracy'):.4f} vs "
        f"{pt('epoch 1', 't-bone', 'inside', 'type_accuracy'):.4f} inside; n {int(pt('epoch 1', 't-bone', 'outside', 'n'))} outside).",
        f"  (4) The S gap is present within every true type for both epochs, while the frame-blind image-centre baseline "
        f"is higher outside ({B['outside'].centre_S:.4f} vs {B['inside'].centre_S:.4f}), so box position alone does not "
        f"account for it; it is consistent with the collision being out of view, but this split cannot separate that "
        f"from the clips being hard, which would need t*-centred windows for the same clips (fold-1 window B is in the "
        f"manifest and cache but has no saved answers; evaluating it needs a model run).",
        f"  (5) The overall inside share {F['all'].share_inside:.4f} is reproduced; fold 1 has the lowest inside share "
        f"({F['1'].share_inside:.4f}, vs {F['0'].share_inside:.4f} fold 0 and {F['2'].share_inside:.4f} fold 2), so "
        f"fold-1 validation numbers mix the two groups in a different proportion than the other folds would.",
    ]


if __name__ == "__main__":
    main()
