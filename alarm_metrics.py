"""Alarm-style metrics for window-level accident detectors.

A detector gives one score per window (window end time t_end). Windows with
score >= tau raise alarms; nearby alarm windows are merged into events.

Per clip (t_star = accident time):
  * false alarm (fa): an event whose time is before t_star - M;
  * hit: any alarm window with t_star - M <= t_end <= t_star + L;
    delay = first such t_end - t_star (can be negative, down to -M);
  * t_neg: negative time in which a false alarm is possible, t_star - M - T_FIRST
    (the first window can only end at T_FIRST).

Aggregated: miss rate MR, delay statistics, timely-detection rate
TDR(d) = (1/K) * count(hit and delay <= d), false alarms per hour FAH, false-alarm
clip rate FAC, and the rule-of-three upper bound on FAH when no false alarm is
observed.

Secondary deployment view only. Frozen: no new metrics.

The constants below are fixed by design and must not be tuned on real data.
This module never reads real_test_set.csv.

Usage:
    py alarm_metrics.py --selftest
"""

import argparse
import math
import sys

import numpy as np

M = 1.0  # ignore margin before t_star, seconds
L = 10.0  # accepted period ends at t_star + L
T_FIRST = 1.875  # span of one window (16 frames at 8 fps = 15/8 s)
MERGE_GAP = 1.0  # merge consecutive alarm windows closer than this
EPS = 1e-9  # float tolerance on time comparisons


def alarm_events(t_end, score, tau, merge_gap=MERGE_GAP):
    """Event times: t_end of the first window of each merged run of alarm windows."""
    t_end = np.asarray(t_end, dtype=float)
    score = np.asarray(score, dtype=float)
    on = np.sort(t_end[score >= tau])
    events = []
    prev = None
    for t in on:
        if prev is None or t - prev > merge_gap + EPS:
            events.append(float(t))
        prev = t
    return events


def clip_outcome(t_end, score, t_star, tau):
    """Outcome of one clip: dict(fa, hit, delay, t_neg)."""
    t_end = np.asarray(t_end, dtype=float)
    score = np.asarray(score, dtype=float)
    lo = t_star - M
    events = alarm_events(t_end, score, tau)
    fa = sum(1 for e in events if e < lo - EPS)
    in_range = (score >= tau) & (t_end >= lo - EPS) & (t_end <= t_star + L + EPS)
    hit = bool(in_range.any())
    delay = float(t_end[in_range].min() - t_star) if hit else None
    return {"fa": fa, "hit": hit, "delay": delay, "t_neg": max(0.0, t_star - M - T_FIRST)}


def aggregate(outcomes, budgets=(0.5, 1.0, 2.0)):
    """Aggregate clip outcomes into MR, delay stats, TDR_d, FAH, FAC, rule-of-three bound."""
    k = len(outcomes)
    if k == 0:
        raise ValueError("aggregate() needs at least one clip outcome")
    delays = np.array([o["delay"] for o in outcomes if o["hit"]], dtype=float)
    fa_total = int(sum(o["fa"] for o in outcomes))
    t_neg_total = float(sum(o["t_neg"] for o in outcomes))
    neg_clips = [o for o in outcomes if o["t_neg"] > 0]
    hours = t_neg_total / 3600.0
    res = {
        "K": k,
        "H": int(len(delays)),
        "MR": 1.0 - len(delays) / k,
        "delay_median": float(np.median(delays)) if len(delays) else None,
        "delay_mean": float(np.mean(delays)) if len(delays) else None,
        "delay_p90": float(np.percentile(delays, 90)) if len(delays) else None,
        "FA_total": fa_total,
        "T_neg_total": t_neg_total,
        "FAH": fa_total / hours if t_neg_total > 0 else None,
        "FAC": (sum(1 for o in neg_clips if o["fa"] >= 1) / len(neg_clips)) if neg_clips else None,
        "rule_of_three_upper": (3.0 / hours) if (fa_total == 0 and t_neg_total > 0) else None,
    }
    for d in budgets:
        res[f"TDR_{d:g}"] = float(np.sum(delays <= d + EPS)) / k
    return res


def amoc_curve(clips, taus):
    """Alarm operating curve. clips: list of (t_end_array, score_array, t_star)."""
    curve = []
    for tau in taus:
        agg = aggregate([clip_outcome(t, s, ts, tau) for t, s, ts in clips])
        curve.append({"tau": float(tau), "FAH": agg["FAH"], "delay_median": agg["delay_median"], "MR": agg["MR"]})
    return curve


def pick_tau_on_validation(clips_val, taus, target_fah):
    """Smallest tau whose FAH <= target_fah, or None if no tau qualifies.

    Threshold selection uses SYNTHETIC validation only; never call it on the real
    test set (that would tune the operating point on test data).
    """
    for row in sorted(amoc_curve(clips_val, taus), key=lambda r: r["tau"]):
        if row["FAH"] is not None and row["FAH"] <= target_fah:
            return row["tau"]
    return None


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest_windows(n_frames, fps, t_star):
    """t_end(j) = j*0.5 + 1.875 while floor(t_end*fps + 0.5) <= N-1; oracle labels."""
    t_end, j = [], 0
    while True:
        t = j * 0.5 + T_FIRST
        if math.floor(t * fps + 0.5) > n_frames - 1:
            break
        t_end.append(t)
        j += 1
    t_end = np.array(t_end)
    label = np.where(t_end < t_star - 1, 0, np.where(t_end < t_star, -1, 1))
    return t_end, label


def selftest():
    tau, tol = 0.5, 1e-3
    clips = {"A": (391, 20, 9.55), "B": (416, 20, 8.65)}
    win = {name: (*_selftest_windows(n, fps, ts), ts) for name, (n, fps, ts) in clips.items()}

    def plus_fa(t, lab, ts):
        s = (lab == 1).astype(float)
        s[3] = 1.0
        return s

    detectors = {
        "always_off": lambda t, lab, ts: np.zeros_like(t),
        "always_on": lambda t, lab, ts: np.ones_like(t),
        "oracle": lambda t, lab, ts: (lab == 1).astype(float),
        "oracle_with_gap": lambda t, lab, ts: (lab != 0).astype(float),
        "late_by_2": lambda t, lab, ts: (t >= ts + 2 - EPS).astype(float),
        "oracle_plus_1_false_alarm": plus_fa,
    }
    # One self-contained dict per detector; FAH = 3600 * FA / T_neg with FA = 2, T_neg = 12.45 s.
    expected = {
        "always_off": {"MR": 1.0, "FA_total": 0, "delay_A": None, "delay_B": None},
        "always_on": {"MR": 0.0, "FA_total": 2, "T_neg_total": 12.45, "FAH": 3600 * 2 / 12.45,
                      "delay_A": -0.675, "delay_B": -0.775},
        "oracle": {"MR": 0.0, "FA_total": 0, "delay_A": 0.325, "delay_B": 0.225, "TDR_0": 0.0, "TDR_1": 1.0},
        "oracle_with_gap": {"MR": 0.0, "FA_total": 0, "delay_A": -0.675, "delay_B": -0.775},
        "late_by_2": {"MR": 0.0, "FA_total": 0, "delay_A": 2.325, "delay_B": 2.225, "TDR_2": 0.0},
        "oracle_plus_1_false_alarm": {"FA_total": 2, "FAH": 3600 * 2 / 12.45},
    }

    print(f"windows: A {len(win['A'][0])} (t_end {win['A'][0][0]}..{win['A'][0][-1]}), "
          f"B {len(win['B'][0])} (t_end {win['B'][0][0]}..{win['B'][0][-1]}); tau={tau}, tol={tol}")
    n_fail = 0
    for name, det in detectors.items():
        outs = {}
        for c, (t, lab, ts) in win.items():
            outs[c] = clip_outcome(t, det(t, lab, ts), ts, tau)
        agg = aggregate(list(outs.values()), budgets=(0, 0.5, 1, 2))
        got = dict(agg, delay_A=outs["A"]["delay"], delay_B=outs["B"]["delay"])
        for key, exp in expected[name].items():
            act = got[key]
            if exp is None or act is None:
                ok = exp is None and act is None
            else:
                ok = abs(act - exp) <= tol
            n_fail += not ok
            act_s = "None" if act is None else f"{act:.4f}"
            exp_s = "None" if exp is None else f"{exp:.4f}"
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<27} {key:<12} actual {act_s:>9}  expected {exp_s:>9}")
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'}")
    return n_fail == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Alarm metrics for window-level accident detectors.")
    ap.add_argument("--selftest", action="store_true", help="run the built-in checks")
    args = ap.parse_args()
    if not args.selftest:
        ap.print_help()
        sys.exit(0)
    sys.exit(0 if selftest() else 1)
