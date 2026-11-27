"""Type model, option 1, STAGE 2 SCORE: score the 755 stage-2 answers as PLAN.txt (+ STAGE2_NOTE.txt additions) and
compare with earlier type models on the same clips. SYNTHETIC ONLY. CPU only, no model. Run with .venv-ml.

Inputs (read only): results/type_vlm8b/stage2_answers.jsonl (must hold all 755 clips), the fold-1 window-A rows of
results/qlora_data/manifest.csv (type, scenario_id, day_night, t_star_inside), metadata-synthetic.csv (true box for the
crop split), results/qlora_train/val_epoch1.csv (QLoRA epoch 1; also its box defines the crop),
results/type_probe_A_map/predictions_oof.csv, results/type_dinov2/predictions_checks.csv,
results/type_dinov2_variants/predictions_checks.csv.

Metrics as type_probe_A_map.py: accuracy; macro accuracy = mean recall over the true types present; per-type recall;
confusion matrix (an invalid answer counts as wrong and has its own column). Scenario bootstrap: scenarios resampled
with replacement, 2,000 draws, numpy default_rng(0), macro over the types present in the draw. Clip bootstrap (added in
STAGE2_NOTE.txt): clips resampled with replacement, 2,000 draws, default_rng(0). Success rule (PLAN.txt): macro
accuracy > 0.20 and the scenario 95% interval (2.5 / 97.5 percentiles) excluding 0.20.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import type_vlm8b as T

OUT = T.OUT
ANSWERS = OUT / "stage2_answers.jsonl"
SESSIONS = OUT / "stage2_sessions.jsonl"
PRED_CSV, MET_JSON = OUT / "stage2_predictions.csv", OUT / "stage2_metrics.json"
CM_COUNTS, CM_ROWNORM, CM_PNG = OUT / "stage2_confusion_counts.csv", OUT / "stage2_confusion_rownorm.csv", OUT / "stage2_confusion.png"
REPORT = OUT / "stage2_report.txt"
R = T.REPO / "results"
TYPES = T.TYPES
TIX = {t: i for i, t in enumerate(TYPES)}
N_BOOT, SEED = 2000, 0
COLS = TYPES + ["invalid"]


def enc(s):
    return s.map(lambda v: TIX.get(v, -1) if isinstance(v, str) else -1).to_numpy()


def metrics(y, p):
    present = sorted(set(y.tolist()))
    rec = {TYPES[t]: float((p[y == t] == t).mean()) for t in present}
    cm = np.array([[int(((y == t) & (p == q)).sum()) for q in list(range(5)) + [-1]] for t in range(5)])
    return {"n": int(len(y)), "accuracy": float((y == p).mean()), "macro_accuracy": float(np.mean(list(rec.values()))),
            "n_types_in_macro": len(present), "recall": rec, "n_per_type": {TYPES[t]: int((y == t).sum()) for t in present},
            "n_invalid": int((p == -1).sum()), "confusion": cm.tolist()}


def macro_of(y, p):
    return float(np.mean([(p[y == t] == t).mean() for t in np.unique(y)]))


def boot_scenario(y, p, scen):
    groups = {s: np.flatnonzero(scen == s) for s in sorted(set(scen))}
    names = list(groups)
    rng = np.random.default_rng(SEED)
    b = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([groups[names[j]] for j in rng.integers(0, len(names), len(names))])
        b[i] = macro_of(y[idx], p[idx])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"unit": "scenario", "n_units": len(names), "n": N_BOOT, "seed": SEED, "ci95": [float(lo), float(hi)],
            "mean": float(b.mean())}


def boot_clip(y, p):
    rng = np.random.default_rng(SEED)
    b = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        b[i] = macro_of(y[idx], p[idx])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"unit": "clip", "n_units": int(len(y)), "n": N_BOOT, "seed": SEED, "ci95": [float(lo), float(hi)],
            "mean": float(b.mean())}


def crop_cover(clips):
    """Where the true (metadata) box falls relative to the crop: full / partly / outside (same code as stage 1)."""
    out = {}
    for p, man, meta, e1 in clips:
        W, H = int(meta["width"]), int(meta["height"])
        left, top, side, _ = T.crop_square(e1, W, H)
        x1, y1, x2, y2 = (float(meta[k]) * s for k, s in (("x1", W), ("y1", H), ("x2", W), ("y2", H)))
        ix = max(0.0, min(x2, left + side) - max(x1, left)) * max(0.0, min(y2, top + side) - max(y1, top))
        a = (x2 - x1) * (y2 - y1)
        out[p] = "fully inside" if ix >= a - 1e-6 else ("outside" if ix <= 0 else "partly inside")
    return out


def load():
    clips = T.load_clips()
    S = [c[0] for c in clips]
    rows = [json.loads(l) for l in ANSWERS.read_text(encoding="utf-8").splitlines() if l.strip()]
    A = pd.DataFrame(rows)
    if len(A) != 755 or set(A["rgb_path"]) != set(S) or A["rgb_path"].duplicated().any():
        T.fail(f"stage2_answers.jsonl must hold each of the 755 clips once (has {len(A)} rows)")
    man = pd.read_csv(R / "qlora_data" / "manifest.csv")
    man = man[(man["fold"] == 1) & (man["window"] == "A")].copy()
    man["rgb_path"] = man["rgb_path"].str.replace("\\", "/", regex=False)
    D = man[["rgb_path", "scenario_id", "type", "weather", "day_night", "t_star_inside"]].merge(A, on="rgb_path", validate="1:1")
    cov = crop_cover(clips)
    D["crop_cover"] = D["rgb_path"].map(cov)
    # comparison predictions on the same clips
    q = pd.read_csv(R / "qlora_train" / "val_epoch1.csv")[["rgb_path", "pred_type"]].rename(columns={"pred_type": "qlora_epoch1"})
    pa = pd.read_csv(R / "type_probe_A_map" / "predictions_oof.csv")
    pa = pa[pa["rgb_path"].isin(S)][["rgb_path", "pred_type", "majority_type"]].rename(
        columns={"pred_type": "probe_A_leave_one_map_out", "majority_type": "majority_leave_one_map_out"})
    d = pd.read_csv(R / "type_dinov2" / "predictions_checks.csv")
    v = pd.read_csv(R / "type_dinov2_variants" / "predictions_checks.csv")
    comp = [q, pa]
    sf = d[(d["split"] == "scenario_folds") & (d["test_group"].astype(str) == "1")]
    comp.append(sf[["rgb_path", "majority_type"]].rename(columns={"majority_type": "majority_scenario_folds"}))
    for name, frame in (("dinov2_224", d), ("dinov2_v1", v[v["variant"] == "v1"]), ("dinov2_v2", v[v["variant"] == "v2"])):
        for split, tag in (("scenario_folds", "scenario_fold1"), ("leave_one_map_out", "leave_one_map_out")):
            x = frame[(frame["split"] == split) & frame["rgb_path"].isin(S)]
            if split == "scenario_folds":
                x = x[x["test_group"].astype(str) == "1"]
            comp.append(x[["rgb_path", "pred_type"]].rename(columns={"pred_type": f"{name}_{tag}"}))
    for c in comp:
        if len(c) != 755 or set(c["rgb_path"]) != set(S):
            T.fail(f"comparison {list(c.columns)[1:]} does not cover the 755 clips exactly")
        D = D.merge(c, on="rgb_path", validate="1:1")
    return D.sort_values("rgb_path").reset_index(drop=True)


MODELS = ["vlm8b_zero_shot", "majority_scenario_folds", "majority_leave_one_map_out", "qlora_epoch1",
          "probe_A_leave_one_map_out", "dinov2_224_scenario_fold1", "dinov2_224_leave_one_map_out",
          "dinov2_v1_scenario_fold1", "dinov2_v1_leave_one_map_out", "dinov2_v2_scenario_fold1", "dinov2_v2_leave_one_map_out"]
SPLITS = {"t_star_inside_window_A": "t_star_inside", "true_box_vs_crop": "crop_cover", "day_night": "day_night"}


def plot(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = np.array(cm, float)
    rn = c / c.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            ax.text(j, i, f"{int(c[i, j])}\n{rn[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if rn[i, j] > 0.55 else "#1a1a1a")
    ax.set_xticks(range(len(COLS)), [f"pred\n{t}" for t in COLS], fontsize=9)
    ax.set_yticks(range(5), [f"true {t} (n={int(c[i].sum())})" for i, t in enumerate(TYPES)], fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(COLS)), minor=True)
    ax.set_yticks(np.arange(-.5, 5), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_title(title, fontsize=10, loc="left")
    fig.text(0.01, 0.01, "cell: count / row-normalised rate (colour = rate); invalid answers count as wrong", fontsize=8,
             color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def f3(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def main():
    T.check_plan()
    T.refuse(PRED_CSV, MET_JSON, CM_COUNTS, CM_ROWNORM, CM_PNG, REPORT)
    D = load()
    D["vlm8b_zero_shot"] = D["final_type"]
    y = enc(D["type"])
    scen = D["scenario_id"].to_numpy()
    res = {"n": len(D), "models": {}, "splits": {}}
    for m in MODELS:
        p = enc(D[m])
        r = metrics(y, p)
        r["bootstrap_scenario"] = boot_scenario(y, p, scen)
        r["bootstrap_clip"] = boot_clip(y, p)
        res["models"][m] = r
    for sname, col in SPLITS.items():
        res["splits"][sname] = {}
        for val in sorted(D[col].astype(str).unique()):
            s = (D[col].astype(str) == val).to_numpy()
            res["splits"][sname][val] = {m: metrics(y[s], enc(D[m])[s]) for m in MODELS}
    V = res["models"]["vlm8b_zero_shot"]
    lo = V["bootstrap_scenario"]["ci95"][0]
    res["success"] = bool(V["macro_accuracy"] > 0.20 and lo > 0.20)
    res["answer_distribution"] = {t: int((D["final_type"] == t).sum()) for t in TYPES}
    res["answer_distribution"]["invalid"] = int((~D["parse_ok"]).sum())
    res["rule_applied"] = int(D["rule_applied"].sum())
    res["parsed_vs_final_differ"] = int((D["parsed_type"].fillna("") != D["final_type"].fillna("")).sum())
    res["invalid_answers"] = D.loc[~D["parse_ok"], ["clip", "raw_answer", "parse_reason"]].to_dict("records")
    res["raw_answer_forms"] = D["raw_answer"].value_counts().head(15).to_dict()
    res["timing"] = {"gen_s_mean": float(D["gen_s"].mean()), "gen_s_max": float(D["gen_s"].max()),
                     "prep_s_mean": float(D["prep_s"].mean()), "sum_prep_gen_h": float((D["gen_s"] + D["prep_s"]).sum() / 3600),
                     "peak_alloc_gb_max": float(D["peak_alloc_gb"].max()), "peak_reserved_gb_max": float(D["peak_reserved_gb"].max()),
                     "new_tokens": D["new_tokens"].value_counts().sort_index().to_dict(),
                     "input_tokens": D["input_tokens"].value_counts().sort_index().to_dict()}
    res["sessions"] = [json.loads(l) for l in SESSIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    # files
    keep = ["rgb_path", "scenario_id", "type", "weather", "day_night", "t_star_inside", "crop_cover", "parse_ok",
            "parsed_type", "final_type"] + MODELS[1:]
    D[keep].to_csv(PRED_CSV, index=False)
    c = np.array(V["confusion"], float)
    idx = [f"true {t}" for t in TYPES]
    pd.DataFrame(c.astype(int), index=idx, columns=[f"pred {t}" for t in COLS]).to_csv(CM_COUNTS)
    pd.DataFrame(c / c.sum(1, keepdims=True), index=idx, columns=[f"pred {t}" for t in COLS]).to_csv(CM_ROWNORM)
    plot(V["confusion"], CM_PNG, f"Zero-shot Qwen3-VL-8B on the crop, 755 fold-1 clips: accuracy "
                                 f"{V['accuracy']:.3f}, macro {V['macro_accuracy']:.3f}")
    MET_JSON.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    write_report(res)


def write_report(res):
    V = res["models"]["vlm8b_zero_shot"]
    bs, bc = V["bootstrap_scenario"], V["bootstrap_clip"]
    tm = res["timing"]
    L = ["TYPE MODEL, OPTION 1 (ZERO-SHOT QWEN3-VL-8B ON A CROP): STAGE 2 REPORT",
         "Synthetic only (755 fold-1 window-A clips); no real file. As PLAN.txt + AMENDMENT_1 (loading) + AMENDMENT_2 step 2 "
         "(embed_tokens and lm_head stored on the CPU; 16 frames at 448 x 448); analyses (a) crop split and (b) clip bootstrap "
         "added in STAGE2_NOTE.txt before any stage-2 answer. PLAN.txt, PROMPT.txt, the amendments and earlier results "
         "unchanged; nothing committed.",
         f"PLAN.txt sha256 {T.PLAN_SHA}; PROMPT.txt sha256 {T.PROMPT_SHA}.", "",
         "1. RUN"]
    for s in res["sessions"]:
        L.append(f"  session {s['start']} -> {s.get('end')}: {s.get('done_this_session')} clips in "
                 f"{s.get('seconds_this_session', 0) / 60:.1f} min (done before {s['done_before']}; total after "
                 f"{s.get('done_total')}); load {s.get('load_s', 0):.1f} s, GPU after load {s.get('mem_after_load_alloc_gb', 0):.2f} GB, "
                 f"4-bit modules {s.get('linear4bit_modules')}; power start '{s.get('power_start')}', end '{s.get('power_end')}'")
    L += [f"  per clip: generate mean {tm['gen_s_mean']:.2f} s (max {tm['gen_s_max']:.2f}), frames mean {tm['prep_s_mean']:.2f} s; "
          f"sum {tm['sum_prep_gen_h']:.2f} h; max peak GPU allocated {tm['peak_alloc_gb_max']:.2f} GB (reserved "
          f"{tm['peak_reserved_gb_max']:.2f} GB); new tokens {tm['new_tokens']}; input tokens {tm['input_tokens']}",
          f"  scene_layout 'not given' for every clip; t-bone -> rear-end rule applied {res['rule_applied']} times "
          f"(parsed and final type differ for {res['parsed_vs_final_differ']} clips).", "",
          "2. SCORE (755 clips; an invalid answer counts as wrong)",
          f"  accuracy {V['accuracy']:.3f}; macro accuracy {V['macro_accuracy']:.3f}",
          "  per-type recall: " + "; ".join(f"{t} {V['recall'][t]:.3f} (n={V['n_per_type'][t]})" for t in V["recall"]),
          "  answer distribution: " + ", ".join(f"{k} {v}" for k, v in res["answer_distribution"].items()),
          f"  invalid answers: {res['answer_distribution']['invalid']}"]
    for r in res["invalid_answers"]:
        L.append(f"    {r['clip']}: {r['parse_reason']}; raw {r['raw_answer']!r}")
    L.append("  most frequent raw answers: " + "; ".join(f"{k!r} x{v}" for k, v in list(res["raw_answer_forms"].items())[:6]))
    L.append("  confusion matrix (rows true, columns predicted; counts / row-normalised):")
    L.append("    " + " " * 16 + "".join(f"{c:>15}" for c in COLS))
    for i, t in enumerate(TYPES):
        row = V["confusion"][i]
        n = sum(row)
        L.append(f"    {'true ' + t:<16}" + "".join(f"{f'{v} / {v / n:.2f}':>15}" for v in row))
    L += ["  files: stage2_confusion_counts.csv, stage2_confusion_rownorm.csv, stage2_confusion.png", "",
          "  COMPARISON on the same 755 clips (macro accuracy with the scenario and clip 95% bootstrap intervals):",
          f"    {'model':<32}{'accuracy':>9}{'macro':>8}   {'scenario CI':<17}{'clip CI':<17}per-type recall " + "/".join(TYPES)]
    for m in MODELS:
        r = res["models"][m]
        L.append(f"    {m:<32}{r['accuracy']:>9.3f}{r['macro_accuracy']:>8.3f}   "
                 f"[{r['bootstrap_scenario']['ci95'][0]:.3f}, {r['bootstrap_scenario']['ci95'][1]:.3f}]   "
                 f"[{r['bootstrap_clip']['ci95'][0]:.3f}, {r['bootstrap_clip']['ci95'][1]:.3f}]   "
                 + " / ".join(f3(r["recall"].get(t)) for t in TYPES))
    L += ["    (majority_scenario_folds = the majority type of folds 0 and 2; majority_leave_one_map_out = the majority type "
          "of the other maps; *_scenario_fold1 = trained on folds 0 and 2; *_leave_one_map_out = trained on the other maps. "
          "A constant answer has macro accuracy 0.20 whenever all 5 types are present.)", "",
          "3. SPLITS (zero-shot model in full; other models: macro accuracy / accuracy)"]
    for sname, d in res["splits"].items():
        L.append(f"  {sname}:")
        for val, mm in d.items():
            r = mm["vlm8b_zero_shot"]
            L.append(f"    {val}: n {r['n']}; accuracy {r['accuracy']:.3f}; macro {r['macro_accuracy']:.3f} over "
                     f"{r['n_types_in_macro']} types; recall " + ", ".join(f"{t} {r['recall'][t]:.3f} (n={r['n_per_type'][t]})"
                                                                          for t in r["recall"]) + f"; invalid {r['n_invalid']}")
            L.append("      others (macro / accuracy): " + "; ".join(
                f"{m} {mm[m]['macro_accuracy']:.3f} / {mm[m]['accuracy']:.3f}" for m in MODELS[1:]))
    L += ["  (Within a split the types are unevenly represented, so the macro accuracies of different split groups average "
          "over different sets and sizes of types; compare models within a group rather than groups with each other.)", "",
          "4. SUCCESS RULE (PLAN.txt): macro accuracy above 0.20 with the 95% scenario bootstrap interval excluding 0.20 "
          f"({bs['n']} draws, seed {bs['seed']})",
          f"  macro accuracy {V['macro_accuracy']:.3f}; scenario bootstrap ({bs['n_units']} scenarios) 95% "
          f"[{bs['ci95'][0]:.3f}, {bs['ci95'][1]:.3f}] (mean {bs['mean']:.3f}); clip bootstrap ({bc['n_units']} clips, added "
          f"in STAGE2_NOTE.txt) 95% [{bc['ci95'][0]:.3f}, {bc['ci95'][1]:.3f}] (mean {bc['mean']:.3f})",
          f"  RESULT: {'SUCCESS' if res['success'] else 'NOT MET'}: macro accuracy "
          f"{'above' if V['macro_accuracy'] > 0.20 else 'not above'} 0.20, and the scenario interval "
          f"{'excludes' if bs['ci95'][0] > 0.20 else 'does not exclude'} 0.20.",
          "  Caveats: fold 1 has only 5 scenarios and each is one type, so the scenario bootstrap mixes scenario and type "
          "variation and is wide; the clip bootstrap treats clips of one scenario as independent and is too narrow. "
          "The model was not trained on synthetic data, but synthetic images differ from real ones: this is a sanity check, "
          "not a prediction of the real score. The scene hint was 'not given' for every clip (no such column), so the "
          "t-bone rule never fired. The clip names and paths in stage2_answers.jsonl contain the true type, but the model "
          "never saw them (it receives only the 16 cropped frames and PROMPT.txt).", "",
          "5. FILES: STAGE2_NOTE.txt; stage2_answers.jsonl (per clip: raw answer, parsed type, final type, parse_ok, "
          "generate time, peak GPU memory, ...; no true types); stage2_sessions.jsonl; stage2_predictions.csv (joined with "
          "the true types and the comparison predictions); stage2_metrics.json; stage2_confusion_*.csv/png; "
          "run_log_stage2_*.txt; scripts type_vlm8b_stage2_run.py, type_vlm8b_stage2_score.py."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
