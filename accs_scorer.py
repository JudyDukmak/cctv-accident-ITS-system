"""ACCS scorer for the ACCIDENT benchmark: temporal (T), spatial (S), type (C), ACCS.

Library only: it reads no data file (no metadata, no real_test_set.csv) and uses no
randomness. Callers pass DataFrames / arrays in.

Sources of every rule
---------------------
* Paper: ACCIDENT, arXiv 2604.09819, Sec. 5, eq. 1-6.
    T_i = exp(-(t*_i - t_hat_i)^2 / (2 sigma_t^2)),  T = mean_i T_i          (eq. 1-2)
    S_i = exp(-[(x*_i - x_hat_i)^2 / (2 sigma_x^2)
                + (y*_i - y_hat_i)^2 / (2 sigma_y^2)]),  S = mean_i S_i      (eq. 3-4)
    C   = Top-1 accuracy over the collision types                           (eq. 5)
    ACCS = 3 / (1/T + 1/S + 1/C)  (harmonic mean)                           (eq. 6)
* Paper footnote 4: sigma_x, sigma_y = mean ground-truth box width and height
  (normalised coordinates).
* Official code ACCIDENT/baselines/heuristic/metrics.py: the T and S formulas above,
  and the reporting set sigma_t in {0.5, 1, 2} s.
* ACCS_SIGMA_T = 1.0: paper Table 5, 3 / (1/0.190 + 1/0.250 + 1/0.335) = 0.245.

Deliberate deviations from metrics.py
-------------------------------------
1. No inner-merge dropping: metrics.py merges predictions with an inner join, so a
   clip without a prediction silently disappears. align() raises instead.
2. One fixed sigma_x / sigma_y given by the caller (computed once with
   spatial_sigmas()), not recomputed from whichever subset is being scored.
3. Spatial multiplier k = 1 (SPATIAL_K) for S and ACCS; metrics.py also prints
   k = 0.5 and k = 2.

Per-class type accuracy = recall of each ground-truth class; macro = mean of the
per-class values over the classes present in the ground truth.

Usage:
    py accs_scorer.py --selftest
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd

TYPES = ["head-on", "rear-end", "sideswipe", "single", "t-bone"]
SIGMAS_T = (0.5, 1.0, 2.0)  # seconds
ACCS_SIGMA_T = 1.0  # paper Table 5: 3/(1/0.190+1/0.250+1/0.335)=0.245
SPATIAL_K = 1.0

GT_COLUMNS = ["path", "accident_time", "center_x", "center_y", "type"]
PRED_COLUMNS = ["path", "t_hat", "x_hat", "y_hat", "type_hat"]


# --------------------------------------------------------------------------- #
# Temporal
# --------------------------------------------------------------------------- #
def temporal_per_clip(t_hat, t_star, sigma_t):
    t_hat = np.asarray(t_hat, dtype=float)
    t_star = np.asarray(t_star, dtype=float)
    return np.exp(-((t_hat - t_star) ** 2) / (2.0 * sigma_t ** 2))


def temporal(t_hat, t_star, sigma_t):
    return float(np.mean(temporal_per_clip(t_hat, t_star, sigma_t)))


# --------------------------------------------------------------------------- #
# Spatial
# --------------------------------------------------------------------------- #
def spatial_sigmas(x1, y1, x2, y2):
    """(mean box width, mean box height) in normalised coordinates (paper footnote 4)."""
    x1, y1, x2, y2 = (np.asarray(v, dtype=float) for v in (x1, y1, x2, y2))
    return float(np.mean(x2 - x1)), float(np.mean(y2 - y1))


def spatial_per_clip(xy_hat, xy_star, sigma_x, sigma_y):
    xy_hat = np.asarray(xy_hat, dtype=float).reshape(-1, 2)
    xy_star = np.asarray(xy_star, dtype=float).reshape(-1, 2)
    dx2 = (xy_hat[:, 0] - xy_star[:, 0]) ** 2 / (2.0 * sigma_x ** 2)
    dy2 = (xy_hat[:, 1] - xy_star[:, 1]) ** 2 / (2.0 * sigma_y ** 2)
    return np.exp(-(dx2 + dy2))


def spatial(xy_hat, xy_star, sigma_x, sigma_y):
    return float(np.mean(spatial_per_clip(xy_hat, xy_star, sigma_x, sigma_y)))


# --------------------------------------------------------------------------- #
# Collision type
# --------------------------------------------------------------------------- #
def _check_labels(labels, name):
    s = pd.Series(list(labels), dtype=object)
    bad = s[~s.isin(TYPES)]
    if len(bad):
        counts = bad.astype(str).value_counts().to_dict()
        print(f"invalid labels in {name}: {counts} (allowed: {TYPES})")
        raise ValueError(f"{name} contains labels not in TYPES: {counts}")


def type_accuracy(y_hat, y_star):
    """(overall Top-1 accuracy, per-class recall dict, macro over classes present in y_star)."""
    y_hat, y_star = list(y_hat), list(y_star)
    if len(y_hat) != len(y_star):
        raise ValueError(f"length mismatch: y_hat {len(y_hat)}, y_star {len(y_star)}")
    _check_labels(y_hat, "y_hat")
    _check_labels(y_star, "y_star")
    yh, ys = np.array(y_hat, dtype=object), np.array(y_star, dtype=object)
    overall = float(np.mean(yh == ys))
    per_class = {}
    for t in TYPES:
        mask = ys == t
        per_class[t] = float(np.mean(yh[mask] == t)) if mask.any() else None
    present = [v for v in per_class.values() if v is not None]
    macro = float(np.mean(present))
    return overall, per_class, macro


# --------------------------------------------------------------------------- #
# ACCS
# --------------------------------------------------------------------------- #
def accs(T, S, C):
    if min(T, S, C) == 0:
        return 0.0
    return 3.0 / (1.0 / T + 1.0 / S + 1.0 / C)


# --------------------------------------------------------------------------- #
# Alignment and full scoring
# --------------------------------------------------------------------------- #
def _basename(p):
    return re.split(r"[\\/]", str(p))[-1]


def align(gt_df, pred_df):
    """Merge ground truth and predictions on the video file name; never drop rows."""
    gt = gt_df.copy()
    pr = pred_df.copy()
    gt["_key"] = gt["path"].map(_basename)
    pr["_key"] = pr["path"].map(_basename)
    for name, d in (("ground truth", gt), ("predictions", pr)):
        dup = d.loc[d["_key"].duplicated(keep=False), "_key"]
        if len(dup):
            raise ValueError(f"{name}: duplicate file names {sorted(set(dup))[:10]} ({dup.nunique()} names)")
    missing = sorted(set(gt["_key"]) - set(pr["_key"]))
    extra = sorted(set(pr["_key"]) - set(gt["_key"]))
    if missing:
        raise ValueError(f"{len(missing)} ground-truth clips have no prediction, e.g. {missing[:10]}")
    if extra:
        raise ValueError(f"{len(extra)} predictions have no ground-truth clip, e.g. {extra[:10]}")
    merged = gt.merge(pr.drop(columns=["path"]), on="_key", how="inner", validate="one_to_one")
    if len(merged) != len(gt):
        raise ValueError(f"alignment produced {len(merged)} rows for {len(gt)} ground-truth clips")
    return merged.drop(columns=["_key"])


def score_all(gt_df, pred_df, sigma_x, sigma_y):
    """All scores for one prediction set. sigma_x, sigma_y are fixed by the caller."""
    for name, d, cols in (("gt_df", gt_df, GT_COLUMNS), ("pred_df", pred_df, PRED_COLUMNS)):
        missing = [c for c in cols if c not in d.columns]
        if missing:
            raise ValueError(f"{name} is missing columns {missing}")
    if not (sigma_x > 0 and sigma_y > 0):
        raise ValueError(f"sigma_x and sigma_y must be > 0, got {sigma_x}, {sigma_y}")
    m = align(gt_df, pred_df)

    t_hat = m["t_hat"].to_numpy(dtype=float)
    if not np.all(np.isfinite(t_hat)):
        raise ValueError(f"{int((~np.isfinite(t_hat)).sum())} predictions have a non-finite t_hat")
    xy_hat = m[["x_hat", "y_hat"]].to_numpy(dtype=float)
    out = ~np.isfinite(xy_hat).all(axis=1) | (xy_hat < 0).any(axis=1) | (xy_hat > 1).any(axis=1)
    if out.any():
        raise ValueError(f"{int(out.sum())} predictions have x_hat/y_hat outside [0, 1] or not finite")

    t_star = m["accident_time"].to_numpy(dtype=float)
    xy_star = m[["center_x", "center_y"]].to_numpy(dtype=float)
    T = {s: temporal(t_hat, t_star, s) for s in SIGMAS_T}
    S = spatial(xy_hat, xy_star, sigma_x * SPATIAL_K, sigma_y * SPATIAL_K)
    C, per_class, macro = type_accuracy(m["type_hat"], m["type"])
    return {
        "N": len(m),
        **{f"T_sigma_{s:g}": T[s] for s in SIGMAS_T},
        "S": S,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "C": C,
        "C_per_class": per_class,
        "C_macro": macro,
        "ACCS": accs(T[ACCS_SIGMA_T], S, C),
        "ACCS_sigma_t": ACCS_SIGMA_T,
        "spatial_k": SPATIAL_K,
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest():
    results = []

    def check(label, actual, expected, tol=1e-3):
        ok = abs(actual - expected) <= tol
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<44} actual {actual:.6g}  expected {expected:.6g}  tol {tol:g}")

    def check_true(label, ok, detail=""):
        results.append(bool(ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<44} {detail}")

    def check_raises(label, fn):
        try:
            with _quiet():
                fn()
        except ValueError as e:
            check_true(label, True, f"raised ValueError: {str(e)[:70]}")
        else:
            check_true(label, False, "did not raise")

    print("a) temporal, t_star = 5.0 (sigma_t = 0.5, 1, 2)")
    table = {
        5.0: [(1, 1e-3), (1, 1e-3), (1, 1e-3)],
        5.5: [(0.607, 1e-3), (0.882, 1e-3), (0.969, 1e-3)],
        6.0: [(0.135, 1e-3), (0.607, 1e-3), (0.882, 1e-3)],
        7.0: [(0.000335, 1e-5), (0.135, 1e-3), (0.607, 1e-3)],
        9.0: [(1.1e-14, 1e-6), (0.000335, 1e-5), (0.135, 1e-3)],
    }
    for t_hat, row in table.items():
        for sigma, (exp, tol) in zip(SIGMAS_T, row):
            check(f"T(t_hat={t_hat}, sigma_t={sigma})", temporal([t_hat], [5.0], sigma), exp, tol)

    print("b) spatial")
    check("S(err=(0.05,0.10), sigma=(0.15,0.20))", spatial([[0.55, 0.60]], [[0.50, 0.50]], 0.15, 0.20), 0.835)

    print("c) ACCS")
    check("accs(0.9, 0.9, 0.05)", accs(0.9, 0.9, 0.05), 0.135)
    check("accs(0.343, 0.50, 0.271)", accs(0.343, 0.50, 0.271), 0.349)
    check("accs(0.190, 0.250, 0.335)", accs(0.190, 0.250, 0.335), 0.245)
    check("accs(0.5, 0.5, 0)", accs(0.5, 0.5, 0), 0.0)

    print("d) type accuracy")
    y_star = ["head-on", "head-on", "head-on", "rear-end"]
    y_hat = ["head-on", "head-on", "head-on", "head-on"]
    overall, per_class, macro = type_accuracy(y_hat, y_star)
    check("overall (3/4)", overall, 0.75)
    check("per_class head-on (3/3)", per_class["head-on"], 1.0)
    check("per_class rear-end (0/1)", per_class["rear-end"], 0.0)
    check_true("per_class absent classes are None",
               all(per_class[t] is None for t in ["sideswipe", "single", "t-bone"]))
    check("macro over present classes (1.0+0.0)/2", macro, 0.5)

    print("e) errors")
    gt = pd.DataFrame({"path": ["real_videos/a_00.mp4", "real_videos\\b_00.mp4"],
                       "accident_time": [1.0, 2.0], "center_x": [0.5, 0.5],
                       "center_y": [0.5, 0.5], "type": ["single", "t-bone"]})
    pred = pd.DataFrame({"path": ["a_00.mp4", "x/b_00.mp4"], "t_hat": [1.0, 2.0],
                         "x_hat": [0.5, 0.5], "y_hat": [0.5, 0.5], "type_hat": ["single", "t-bone"]})
    check_true("align matches '\\' and '/' paths by file name", len(align(gt, pred)) == 2)
    check_raises("align raises on a missing prediction", lambda: align(gt, pred.iloc[:1]))
    extra = pd.concat([pred, pred.iloc[:1].assign(path="c_00.mp4")], ignore_index=True)
    check_raises("align raises on an extra prediction", lambda: align(gt, extra))
    dup = pd.concat([pred, pred.iloc[:1].assign(path="other/a_00.mp4")], ignore_index=True)
    check_raises("align raises on a duplicate basename", lambda: align(gt, dup))
    check_raises("type_accuracy raises on 'single-vehicle'",
                 lambda: type_accuracy(["single-vehicle", "t-bone"], ["single", "t-bone"]))

    print("f) score_all end to end (extra)")
    r = score_all(gt, pred, 0.1, 0.1)
    check("perfect predictions: ACCS", r["ACCS"], 1.0)
    check_raises("score_all raises on x_hat > 1", lambda: score_all(gt, pred.assign(x_hat=[0.5, 1.2]), 0.1, 0.1))
    check_raises("score_all raises on non-finite t_hat",
                 lambda: score_all(gt, pred.assign(t_hat=[1.0, np.nan]), 0.1, 0.1))

    n_fail = results.count(False)
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} ({len(results)} checks)")
    return n_fail == 0


class _quiet:
    """Silence the diagnostic print of _check_labels inside expected-error checks."""

    def __enter__(self):
        import io
        self._old, sys.stdout = sys.stdout, io.StringIO()

    def __exit__(self, *exc):
        sys.stdout = self._old
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ACCS scorer (library). Use --selftest to run the checks.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if not args.selftest:
        ap.print_help()
        sys.exit(0)
    sys.exit(0 if selftest() else 1)
