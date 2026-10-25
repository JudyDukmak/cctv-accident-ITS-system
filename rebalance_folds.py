"""Compare the current synthetic fold assignment with a random-search rebalance.

READ-ONLY, SYNTHETIC DATA ONLY. Reads synthetic_check/scenario_table.csv and
synthetic_split.csv. Never opens real data and never writes synthetic_split.csv.

cost(a) for an assignment a (fold 0/1/2 per scenario):
    for each fold k: |n_clips - 2211/3| / 100                      (size)
                   + |median scenario accident_time - global_median| (timing)
                   + 5 * (5 - number of collision types)            (type penalty)

Random search, seed 0, 200_000 draws, skipping draws that do not use all 3 folds:
    run 1 = scenario split (no map constraint);
    run 2 = map split (same seed; also skip draws where a map spans several folds).

Output: synthetic_check/rebalance_report.txt.

Usage:
    py rebalance_folds.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SCENARIO_TABLE = REPO_ROOT / "synthetic_check" / "scenario_table.csv"
SPLIT_CSV = REPO_ROOT / "synthetic_split.csv"
OUT_TXT = REPO_ROOT / "synthetic_check" / "rebalance_report.txt"
TOTAL_CLIPS = 2211
N_FOLDS = 3
N_TYPES = 5
SEED = 0
N_ITER = 200_000


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))


def fail(msg):
    sys.exit(f"ERROR: {msg}")


# --------------------------------------------------------------------------- #
# Cost, written as in the evaluation document (pandas), plus its three parts
# --------------------------------------------------------------------------- #
def cost_parts(scen, a, global_median):
    size = timing = types = 0.0
    for k in range(N_FOLDS):
        s = scen[a == k]
        size += abs(s.n_clips.sum() - TOTAL_CLIPS / 3) / 100
        timing += abs(np.median(s.accident_time) - global_median)
        types += 5 * (N_TYPES - s.ctype.nunique())
    return {"size": size, "timing": timing, "types": types, "total": size + timing + types}


def cost(scen, a, global_median):
    return cost_parts(scen, a, global_median)["total"]


def maps_shared_across_folds(scen, a):
    return bool((pd.Series(a, index=scen.index).groupby(scen["map"]).nunique() > 1).any())


# --------------------------------------------------------------------------- #
# Same cost on plain arrays, for the 2 x 200_000 search loop
# --------------------------------------------------------------------------- #
class FastCost:
    def __init__(self, scen, global_median):
        self.n = scen["n_clips"].to_numpy(dtype=float)
        self.t = scen["accident_time"].to_numpy(dtype=float)
        self.c = pd.factorize(scen["ctype"])[0]
        self.m = pd.factorize(scen["map"])[0]
        self.g = global_median

    def __call__(self, a):
        total = 0.0
        for k in range(N_FOLDS):
            mask = a == k
            total += abs(self.n[mask].sum() - TOTAL_CLIPS / 3) / 100
            total += abs(np.median(self.t[mask]) - self.g)
            total += 5 * (N_TYPES - len(set(self.c[mask].tolist())))
        return total

    def maps_shared(self, a):
        for mp in set(self.m.tolist()):
            if len(set(a[self.m == mp].tolist())) > 1:
                return True
        return False


def search(fast, n_scen, map_split):
    rng = np.random.default_rng(SEED)
    best_a, best_cost, n_valid = None, np.inf, 0
    for _ in range(N_ITER):
        a = rng.integers(0, 3, size=n_scen)
        if len(set(a.tolist())) < 3:
            continue
        if map_split and fast.maps_shared(a):
            continue
        n_valid += 1
        c = fast(a)
        if c < best_cost:
            best_a, best_cost = a.copy(), c
    return best_a, best_cost, n_valid


def partition(scen, a):
    return frozenset(frozenset(scen.loc[a == k, "scenario_id"]) for k in range(N_FOLDS))


def describe(r, scen, a, global_median, title):
    p = cost_parts(scen, a, global_median)
    r(f"== {title}")
    r(f"   assignment (scenario order above): {a.tolist()}")
    r(f"   total cost {p['total']:.4f} = size {p['size']:.4f} + timing {p['timing']:.4f} "
      f"+ type penalty {p['types']:.4f}")
    r(f"   maps shared across folds: {maps_shared_across_folds(scen, a)}")
    for k in range(N_FOLDS):
        s = scen[a == k]
        r(f"   fold {k}: n_clips {int(s.n_clips.sum())}, median scenario accident_time "
          f"{np.median(s.accident_time):.3f}s, types {sorted(s.ctype.unique())}")
        r(f"           scenarios: {', '.join(s.scenario_id)}")
    r("")
    return p


def main():
    for f in (SCENARIO_TABLE, SPLIT_CSV):
        if not f.is_file():
            fail(f"{f} not found")
    table = pd.read_csv(SCENARIO_TABLE)
    split = pd.read_csv(SPLIT_CSV, usecols=["scenario_id", "val_fold"])

    scen = pd.DataFrame({
        "scenario_id": table["scenario_id"],
        "map": table["map"],
        "ctype": table["collision_type"],
        "n_clips": table["n_clips"].astype(int),
        "accident_time": table["median_accident_time_s"].astype(float),
    }).reset_index(drop=True)
    if scen["scenario_id"].duplicated().any() or len(scen) != 14:
        fail(f"expected 14 unique scenarios, got {len(scen)}")
    if int(scen.n_clips.sum()) != TOTAL_CLIPS:
        fail(f"n_clips sums to {int(scen.n_clips.sum())}, expected {TOTAL_CLIPS}")
    folds = split.groupby("scenario_id")["val_fold"].agg(["nunique", "first"])
    if (folds["nunique"] > 1).any() or set(folds.index) != set(scen.scenario_id):
        fail("synthetic_split.csv does not give exactly one fold per scenario in scenario_table.csv")
    current = scen["scenario_id"].map(folds["first"]).to_numpy(dtype=int)

    global_median = float(np.median(scen.accident_time))
    fast = FastCost(scen, global_median)

    r = Report()
    r("Fold rebalance report (synthetic only)")
    r(f"inputs: {SCENARIO_TABLE.name}, {SPLIT_CSV.name}; seed {SEED}, {N_ITER} draws per run")
    r("scenario order (index -> scenario_id, map, ctype, n_clips, accident_time):")
    for i, row in scen.iterrows():
        r(f"  {i:>2} {row['scenario_id']:<17} {row['map']:<9} {row['ctype']:<10} "
          f"{row['n_clips']:>4} {row['accident_time']:6.2f}")
    r(f"global_median (median of the 14 scenario medians): {global_median:.4f}s")
    r("")

    best1, cost1, valid1 = search(fast, len(scen), map_split=False)
    best2, cost2, valid2 = search(fast, len(scen), map_split=True)

    # The fast loop must reproduce the reference cost exactly.
    for name, a in (("current", current), ("run 1 best", best1), ("run 2 best", best2)):
        if a is None:
            continue
        ref, quick = cost(scen, a, global_median), fast(a)
        if abs(ref - quick) > 1e-9:
            fail(f"fast cost {quick} != reference cost {ref} for {name}")

    describe(r, scen, current, global_median, "CURRENT split (val_fold from synthetic_split.csv)")
    describe(r, scen, best1, global_median, f"RUN 1 best - scenario split ({valid1} valid assignments)")
    if best2 is None:
        r("== RUN 2 best - map split: no valid assignment found")
        r("")
    else:
        describe(r, scen, best2, global_median, f"RUN 2 best - map split ({valid2} valid assignments)")
    r(f"Run 2 evaluated {valid2} valid assignments (of {N_ITER} draws).")

    cur = partition(scen, current)
    r(f"current split == run 1 best (as partitions): {cur == partition(scen, best1)}")
    r(f"current split == run 2 best (as partitions): "
      f"{best2 is not None and cur == partition(scen, best2)}")

    OUT_TXT.write_text("\n".join(r.lines) + "\n", encoding="ascii")
    print(f"\nsaved {OUT_TXT}")


if __name__ == "__main__":
    main()
