"""Alternative window labelling schemes A / B / C for the synthetic windows.

SYNTHETIC DATA ONLY, no randomness. Reads synthetic_windows.csv, split_manifest.json and
windows_config.json (all read-only) and writes synthetic_window_labels.csv.

Per window (t_star = accident time, margin = 1.0 s, span = 15/8 s):
    t_centre = t_start + 0.9375                              (half the window span)
    label_A  = existing "label" column, copied unchanged     (0 pre / -1 gap / 1 onset+post)
    label_B  = 0   if t_end < t_star - 1.0
               1   if t_start <= t_star <= t_end             (window contains the impact)
              -1   otherwise (margin or aftermath; ignored in the loss)
    weight_C_text = 0.0 if t_end < t_star - 1.0
                    1.0 if t_start <= t_star <= t_end
                    exp(-d^2 / (2 * 0.5^2)) otherwise, d = min(|t_star - t_start|, |t_star - t_end|)
    weight_C_code = 1.0 if t_start <= t_star <= t_end
                    exp(-d^2 / (2 * 0.5^2)) otherwise (same d; no pre-window rule)

weight_C_text follows the Section 6.3 text plus code: the text says "In all schemes,
windows ending before t* - 1 s are normal", so those windows get 0.
weight_C_code follows the Section 6.3 label_C code literally: it applies the Gaussian
everywhere, so a window ending exactly at t* - 1 s gets exp(-2) = 0.135 (and smaller
non-zero values further before t*). The evaluation file is inconsistent here; the
choice between the two will be confirmed with the supervisor.

"t_end < t_star - 1.0" is evaluated as t_end < round(t_star - 1.0, 6), the same rounding
make_windows.py uses, so label_B == 0 exactly where label_A == 0. The containment test
t_start <= t_star <= t_end uses the same 6-decimal rounded times.

The manifest is updated (window_labels_sha256) only with --record-hash.

Usage:
    py make_window_labels.py [--record-hash] [--overwrite]
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
WINDOWS_CSV = REPO_ROOT / "synthetic_windows.csv"
MANIFEST = REPO_ROOT / "split_manifest.json"
CONFIG = REPO_ROOT / "windows_config.json"
OUT_CSV = REPO_ROOT / "synthetic_window_labels.csv"
OUT_SUMMARY = REPO_ROOT / "synthetic_check" / "window_labels_summary.txt"

HALF_SPAN = 0.9375  # (16 - 1) / 8 / 2 seconds
MARGIN = 1.0  # seconds
SIGMA_C = 0.5  # seconds
TIME_DECIMALS = 6
OUT_COLUMNS = ["rgb_path", "scenario_id", "val_fold", "window_idx", "t_start", "t_end", "t_centre",
               "t_star", "label_A", "label_B", "weight_C_text", "weight_C_code", "split_version"]
TIMELINE_CLIP = "Town05_sideswipe_rain_44"
EXPECTED_B = {**{j: 0 for j in range(0, 14)}, **{j: 1 for j in range(16, 20)},
              14: -1, 15: -1, **{j: -1 for j in range(20, 36)}}
EXPECTED_CENTRE = {16: 8.9375}
EXPECTED_C = {13: 0.0, 14: 0.4020, 15: 0.9406, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0,
              20: 0.6670, 21: 0.1645, 22: 0.0149}  # weight_C_text
EXPECTED_C_CODE = {12: 0.0037, 13: 0.0632}  # weight_C_code; windows 14-35 must equal weight_C_text
CODE_EQUALS_TEXT_FROM = 14
TOL_C = 0.0005


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    if not path.is_file():
        fail(f"{path} not found")
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, ValueError) as e:
        fail(f"cannot read {path}: {e}")


def check_inputs():
    manifest = read_json(MANIFEST)
    config = read_json(CONFIG)
    if not WINDOWS_CSV.is_file():
        fail(f"{WINDOWS_CSV} not found")
    expected = manifest.get("windows_sha256")
    got = sha256_file(WINDOWS_CSV)
    if got != expected:
        fail(f"{WINDOWS_CSV.name} SHA-256 {got} does not match windows_sha256 {expected} in {MANIFEST.name}")
    span = (config["window_frames"] - 1) / config["model_fps"]
    if not math.isclose(span / 2, HALF_SPAN) or not math.isclose(config["ignore_margin_s"], MARGIN):
        fail(f"windows_config.json (half span {span / 2}, margin {config['ignore_margin_s']}) does not match "
             f"HALF_SPAN {HALF_SPAN} / MARGIN {MARGIN}")
    return manifest, got


def compute_labels(win):
    ts = np.round(win["t_start"].to_numpy(float), TIME_DECIMALS)
    te = np.round(win["t_end"].to_numpy(float), TIME_DECIMALS)
    tstar = np.round(win["t_star"].to_numpy(float), TIME_DECIMALS)
    pre = te < np.round(tstar - MARGIN, TIME_DECIMALS)
    contains = (ts <= tstar) & (tstar <= te)
    label_b = np.where(pre, 0, np.where(contains, 1, -1))
    d = np.minimum(np.abs(tstar - ts), np.abs(tstar - te))
    gauss = np.exp(-d ** 2 / (2 * SIGMA_C ** 2))
    weight_c_code = np.where(contains, 1.0, gauss)  # Section 6.3 code, literally
    weight_c_text = np.where(pre, 0.0, weight_c_code)  # plus the text rule for pre windows
    out = pd.DataFrame({
        "rgb_path": win["rgb_path"],
        "scenario_id": win["scenario_id"],
        "val_fold": win["val_fold"],
        "window_idx": win["window_idx"],
        "t_start": win["t_start"],
        "t_end": win["t_end"],
        "t_centre": np.round(ts + HALF_SPAN, TIME_DECIMALS),
        "t_star": win["t_star"],
        "label_A": win["label"],
        "label_B": label_b,
        "weight_C_text": weight_c_text,
        "weight_C_code": weight_c_code,
        "split_version": win["split_version"],
    })
    return out[OUT_COLUMNS]


def summarize(lab, win):
    r = Report()
    r("1. label_B counts")
    r("   overall: " + ", ".join(f"{k}: {int((lab['label_B'] == k).sum())}" for k in (-1, 0, 1)))
    r(pd.crosstab(lab["val_fold"], lab["label_B"], margins=True).to_string())

    r("")
    mism = int(((lab["label_B"] == 0) != (lab["label_A"] == 0)).sum())
    r(f"2. label_B == 0 exactly where label_A == 0: mismatches {mism} -> {'OK' if mism == 0 else 'MISMATCH'}")

    n_b1 = int((lab["label_B"] == 1).sum())
    onset = win["phase"] == "onset"
    start_eq = win["t_start"] == win["t_star"]
    n_expected = int((onset | start_eq).sum())
    same_rows = bool(((lab["label_B"] == 1) == (onset | start_eq)).all())
    r(f"3. label_B == 1 windows: {n_b1}; onset windows {int(onset.sum())} + windows with t_start == t_star "
      f"{int(start_eq.sum())} = {n_expected}; same rows: {same_rows} -> "
      f"{'OK' if same_rows and n_b1 == n_expected else 'MISMATCH'}")

    pos = lab[lab["label_B"] == 1].groupby("rgb_path").size().reindex(lab["rgb_path"].unique(), fill_value=0)
    r(f"4. positives per clip under B: min {pos.min()} / median {pos.median():g} / max {pos.max()}; "
      f"clips with zero positives: {int((pos == 0).sum())} -> {'OK' if (pos == 0).sum() == 0 else 'MISMATCH'}")

    for col in ("weight_C_text", "weight_C_code"):
        w = lab[col]
        r(f"5. {col}: exactly 1.0: {int((w == 1.0).sum())}, exactly 0.0: {int((w == 0.0).sum())}, "
          f"strictly between: {int(((w > 0) & (w < 1)).sum())}")
    differ = lab["weight_C_text"] != lab["weight_C_code"]
    n_pre = int((lab["label_B"] == 0).sum())
    only_pre = bool((lab.loc[differ, "label_B"] == 0).all())
    diff_ok = only_pre and int(differ.sum()) <= n_pre
    r(f"   weight_C_text != weight_C_code: {int(differ.sum())} windows; all have t_end < t_star - 1 "
      f"(label_B == 0): {only_pre}; pre windows: {n_pre} -> {'OK' if diff_ok else 'MISMATCH'}")
    r(f"   pre windows where both weights are 0 (Gaussian underflow or exact 0): "
      f"{int(((lab['label_B'] == 0) & ~differ).sum())}")

    r("")
    clip_mask = lab["rgb_path"].str.contains(TIMELINE_CLIP + ".mp4", regex=False)
    tl = lab[clip_mask]
    r(f"6. Timeline of {TIMELINE_CLIP}" + (f" (t_star {tl['t_star'].iloc[0]})" if len(tl) else ""))
    if tl.empty:
        r("   clip not found")
    else:
        r(tl[["window_idx", "t_start", "t_end", "t_centre", "label_A", "label_B", "weight_C_text",
              "weight_C_code"]]
          .to_string(index=False, formatters={"weight_C_text": "{:.4f}".format,
                                              "weight_C_code": "{:.4f}".format}))

    r("")
    r(f"7. EXPECTED checks for {TIMELINE_CLIP}")
    n_bad = 0
    by_j = tl.set_index("window_idx")

    def line(label, ok, got, exp):
        nonlocal n_bad
        n_bad += not ok
        r(f"   {'OK      ' if ok else 'MISMATCH'} {label}: got {got}, expected {exp}")

    if len(by_j) != len(EXPECTED_B):
        line("number of windows", False, len(by_j), len(EXPECTED_B))
    for j, e in sorted(EXPECTED_B.items()):
        g = int(by_j.at[j, "label_B"]) if j in by_j.index else None
        line(f"window {j:>2} label_B", g == e, g, e)
    for j, e in EXPECTED_CENTRE.items():
        g = float(by_j.at[j, "t_centre"]) if j in by_j.index else None
        line(f"window {j:>2} t_centre", g is not None and abs(g - e) < 1e-9, g, e)
    for j, e in EXPECTED_C.items():
        g = float(by_j.at[j, "weight_C_text"]) if j in by_j.index else None
        line(f"window {j:>2} weight_C_text (tol {TOL_C})", g is not None and abs(g - e) <= TOL_C,
             None if g is None else round(g, 4), e)
    for j, e in EXPECTED_C_CODE.items():
        g = float(by_j.at[j, "weight_C_code"]) if j in by_j.index else None
        line(f"window {j:>2} weight_C_code (tol {TOL_C})", g is not None and abs(g - e) <= TOL_C,
             None if g is None else round(g, 4), e)
    tail = by_j.loc[by_j.index >= CODE_EQUALS_TEXT_FROM]
    n_tail_diff = int((tail["weight_C_code"] - tail["weight_C_text"]).abs().gt(TOL_C).sum())
    line(f"windows {CODE_EQUALS_TEXT_FROM}-{int(by_j.index.max())} weight_C_code == weight_C_text "
         f"(windows differing)", n_tail_diff == 0, n_tail_diff, 0)
    return r, n_bad + (mism != 0) + (not same_rows) + int((pos == 0).sum() != 0) + (not diff_ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record-hash", action="store_true",
                    help="store window_labels_sha256 in split_manifest.json")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing synthetic_window_labels.csv whose content differs")
    args = ap.parse_args()

    manifest, windows_sha = check_inputs()
    win = pd.read_csv(WINDOWS_CSV)
    if not win[["rgb_path", "window_idx"]].equals(
            win.sort_values(["rgb_path", "window_idx"], kind="mergesort")[["rgb_path", "window_idx"]]
            .reset_index(drop=True)):
        fail(f"{WINDOWS_CSV.name} is not sorted by (rgb_path, window_idx)")
    spans = (win["t_end"] - win["t_start"]).round(TIME_DECIMALS)
    if not np.allclose(spans, 2 * HALF_SPAN):
        fail(f"window spans differ from {2 * HALF_SPAN} s: {sorted(spans.unique())[:5]}")

    lab = compute_labels(win)
    text = lab.to_csv(index=False, lineterminator="\n")
    if OUT_CSV.exists() and OUT_CSV.read_text(encoding="ascii") == text:
        print(f"{OUT_CSV.name} already exists with identical content (unchanged).")
    elif OUT_CSV.exists() and not args.overwrite:
        fail(f"{OUT_CSV} exists with different content; rerun with --overwrite to replace it")
    else:
        OUT_CSV.write_text(text, encoding="ascii", newline="")
        print(f"wrote {OUT_CSV}")
    labels_sha = sha256_file(OUT_CSV)

    if args.record_hash:
        before = read_json(MANIFEST)
        after = dict(before, window_labels_sha256=labels_sha)
        MANIFEST.write_text(json.dumps(after, indent=2) + "\n", encoding="ascii", newline="\n")
        reread = read_json(MANIFEST)
        changed = [k for k in set(before) | set(reread)
                   if k != "window_labels_sha256" and before.get(k) != reread.get(k)]
        if changed:
            fail(f"manifest update changed other keys: {changed}")
        print(f"{MANIFEST.name}: window_labels_sha256 = {labels_sha}")
    else:
        print(f"window_labels_sha256 = {labels_sha} (manifest unchanged; pass --record-hash to store it)")

    r, n_bad = summarize(lab, win)
    header = (f"Synthetic window labels A / B / C\nwindows_sha256: {windows_sha} (matches manifest)\n"
              f"split_version: {', '.join(sorted(lab['split_version'].unique()))}\n"
              f"window_labels_sha256: {labels_sha}\nrows: {len(lab)}\n")
    print()
    OUT_SUMMARY.parent.mkdir(exist_ok=True)
    OUT_SUMMARY.write_text(header + "\n" + "\n".join(r.lines) + "\n", encoding="ascii", newline="\n")
    print(header)
    print(f"saved {OUT_SUMMARY}")
    if n_bad:
        fail(f"{n_bad} check(s) failed (see MISMATCH lines above)")


if __name__ == "__main__":
    main()
