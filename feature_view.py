"""Look at the saved type-probe features (CPU only; no model, no GPU, no new features; synthetic only).

Reads cache/type_probe_A/features_fold<k>.npz and results/type_probe_A/windows.csv; changes neither. Writes only to
results/feature_view/ and refuses to overwrite. Run with .venv-ml (numpy, pandas, matplotlib; no scikit-learn, so PCA
is an SVD and the silhouette score is computed directly from the pairwise Euclidean distances).

  1. description.txt   the saved files: names, format, arrays, shapes, rows / columns, fold model and window per row,
                       first 5 numbers of 3 rows
  2. pca_variance.csv  fold model 1, t_hat window of each clip (2211 rows, mean of original and flipped features),
                       standardised; share of variance of the first 10 components; pca_2d.csv the 2-D points
  3. pca_map_type.png  the same 2-D points, coloured by map (left) and by collision type (right)
  4. silhouette.csv    silhouette score of the standardised 768-number points grouped by map, scenario and type
                       (all 2211 clips; also the 755 fold-1 clips, which fold model 1 did not train on); a reference
                       with the labels shuffled (numpy default_rng(0)) is listed next to each
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
CACHE = REPO / "cache" / "type_probe_A"
WINDOWS = REPO / "results" / "type_probe_A" / "windows.csv"
OUT = REPO / "results" / "feature_view"
FOLD_MODEL = 1
MAPS = ["Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
TYPES = ["head-on", "rear-end", "sideswipe", "single", "t-bone"]
# Colours as given by the user (in this order); every category also has its own marker shape and a direct label at
# its centroid.
COLOURS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#000000"]  # red, blue, green, orange, purple, black
MAP_COL = COLOURS[:6]
TYPE_COL = COLOURS[:5]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def describe(w):
    L = ["SAVED TYPE-PROBE FEATURES: cache/type_probe_A/ (written by type_probe_A.py --extract, 2026-11-25)", ""]
    pairs = {}
    for _, r in w.iterrows():
        pairs.setdefault((r["rgb_path"], int(r["w_tstar"])), set()).add("t*")
        pairs.setdefault((r["rgb_path"], int(r["w_that"])), set()).add("t_hat")
    for k in (0, 1, 2):
        p = CACHE / f"features_fold{k}.npz"
        z = np.load(p, allow_pickle=True)  # rgb_path is stored as an object array (our own file)
        L.append(f"{p.name}: numpy .npz (uncompressed, np.savez), {p.stat().st_size} bytes; features from Classifier fold "
                 f"model {k} (runs/C_cctv_seed0/fold_{k}/checkpoints/best.pt)")
        for name in z.files:
            a = z[name]
            L.append(f"  {name:10s} dtype {str(a.dtype):7s} shape {a.shape}")
        roles = [pairs.get((r, int(i)), set()) for r, i in zip(z["rgb_path"], z["window_idx"])]
        n_s = sum(r == {"t*"} for r in roles)
        n_h = sum(r == {"t_hat"} for r in roles)
        n_b = sum(r == {"t*", "t_hat"} for r in roles)
        L.append(f"  rows: one per (clip, Classifier window), {len(roles)} rows, sorted by rgb_path then window_idx; "
                 f"{len(set(z['rgb_path']))} clips; each row is the t* window of its clip ({n_s}), the frozen out-of-fold "
                 f"t_hat window ({n_h}), or both ({n_b}); {sum(not r for r in roles)} rows match neither")
        if k == 0:
            L += ["  rgb_path[i]   the clip (synthetic video path); window_idx[i] the Classifier window (index in "
                  "synthetic_window_labels.csv; 16 frames at 8 fps from t_start = 0.5 * window_idx s)",
                  "  feat[i, j, :] 768 numbers = the input of the Classifier's head, fc_norm(mean over tokens of the last "
                  "hidden states); j = 0 original frames, j = 1 horizontally flipped frames; columns 0..767 are the "
                  "feature dimensions (no names)",
                  "  logit[i, j]   the Classifier's window logit (accident score before the sigmoid) for the same "
                  "window and flip"]
        for i in range(3):
            L.append(f"  row {i}: {z['rgb_path'][i].split('/')[-1]} window {int(z['window_idx'][i])} "
                     f"({'+'.join(sorted(roles[i]))}); feat[{i},0,:5] = "
                     + ", ".join(f"{v:.6f}" for v in z["feat"][i, 0, :5]) + f"; feat[{i},1,:5] = "
                     + ", ".join(f"{v:.6f}" for v in z["feat"][i, 1, :5]) + f"; logit {z['logit'][i].tolist()}")
        L.append("")
    L.append("The three files hold the same rows (same clips and windows, same order); only the fold model differs. For "
             "clips in fold k, fold model k is out of sample; for the other clips it was trained on them.")
    return L


def silhouette(D, lab):
    """Mean silhouette (points in a singleton group score 0, as scikit-learn)."""
    lab = np.asarray(lab)
    groups = {g: np.flatnonzero(lab == g) for g in np.unique(lab)}
    s = np.zeros(len(lab))
    for g, idx in groups.items():
        if len(idx) < 2:
            continue
        a = D[np.ix_(idx, idx)].sum(1) / (len(idx) - 1)
        b = np.min([D[np.ix_(idx, o)].mean(1) for h, o in groups.items() if h != g], axis=0)
        s[idx] = (b - a) / np.maximum(a, b)
    return float(s.mean())


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    outs = [OUT / n for n in ("description.txt", "pca_variance.csv", "pca_2d.csv", "pca_map_type.png", "silhouette.csv")]
    for p in outs:
        if p.exists():
            fail(f"{p} exists; refusing to overwrite")
    w = pd.read_csv(WINDOWS)
    w["map"] = w["scenario_id"].str.split("_").str[0]
    if len(w) != 2211:
        fail("expected 2211 clips")
    # 1
    D_ = describe(w)
    outs[0].write_text("\n".join(D_) + "\n", encoding="utf-8")
    print("\n".join(D_))
    # 2: fold model 1, t_hat window, mean of original and flipped
    z = np.load(CACHE / f"features_fold{FOLD_MODEL}.npz", allow_pickle=True)
    key = {(r, int(i)): j for j, (r, i) in enumerate(zip(z["rgb_path"], z["window_idx"]))}
    X = z["feat"].astype(np.float64).mean(1)[[key[(r, i)] for r, i in zip(w["rgb_path"], w["w_that"])]]
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - X.mean(0)) / sd
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    pv = pd.DataFrame({"component": np.arange(1, 11), "variance_share": var[:10], "cumulative": np.cumsum(var)[:10]})
    pv.to_csv(outs[1], index=False)
    print(f"\nPCA (fold model {FOLD_MODEL}, t_hat windows, {len(Xs)} clips x {Xs.shape[1]} standardised features):")
    print(pv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    P2 = Xs @ Vt[:2].T
    pd.DataFrame({"rgb_path": w["rgb_path"], "map": w["map"], "scenario_id": w["scenario_id"], "type": w["type"],
                  "fold": w["fold"], "pc1": P2[:, 0], "pc2": P2[:, 1]}).to_csv(outs[2], index=False)
    # 3: figure
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, (col, cats, cols, title) in zip(axes, [("map", MAPS, MAP_COL, "coloured by map (6)"),
                                                   ("type", TYPES, TYPE_COL, "coloured by collision type (5)")]):
        lab = w[col].to_numpy()
        for c, colr, mk in zip(cats, cols, MARKERS):
            m = lab == c
            ax.scatter(P2[m, 0], P2[m, 1], s=14, c=colr, marker=mk, alpha=0.55, linewidths=0.4, edgecolors=SURF,
                       label=f"{c} (n {int(m.sum())})")
        for c in cats:  # direct label at the centroid, in ink
            m = lab == c
            cx, cy = P2[m, 0].mean(), P2[m, 1].mean()
            ax.annotate(c, (cx, cy), fontsize=9, color=INK, ha="center", va="center", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=INK2, lw=0.6, alpha=0.85))
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"PC1 ({var[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} of variance)")
        ax.set_title(title, color=INK)
        ax.legend(loc="upper right", fontsize=8, markerscale=1.6, labelcolor=INK)
    fig.suptitle(f"Classifier features (C_cctv seed 0, fold model {FOLD_MODEL}, t_hat window, 2,211 synthetic clips), "
                 "standardised, PCA to 2-D; labels mark group centroids", color=INK)
    fig.savefig(outs[3], dpi=150)
    plt.close(fig)
    # 4: silhouette on the full standardised 768 numbers
    G = Xs @ Xs.T
    sq = np.diag(G)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0))
    np.fill_diagonal(D, 0.0)
    rng = np.random.default_rng(0)
    rows = []
    f1 = (w["fold"] == FOLD_MODEL).to_numpy()
    for scope, m in (("all 2211 clips", np.ones(len(w), bool)), (f"fold-{FOLD_MODEL} clips (encoder out of sample)", f1)):
        Dm = D[np.ix_(m, m)]
        for by in ("map", "scenario_id", "type"):
            lab = w.loc[m, by].to_numpy()
            rows.append({"scope": scope, "grouped_by": by, "n_points": int(m.sum()), "n_groups": len(set(lab)),
                         "silhouette": silhouette(Dm, lab), "silhouette_labels_shuffled": silhouette(Dm, rng.permutation(lab))})
    sil = pd.DataFrame(rows)
    sil.to_csv(outs[4], index=False)
    print("\nSilhouette (standardised 768 numbers, Euclidean):")
    print(sil.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
