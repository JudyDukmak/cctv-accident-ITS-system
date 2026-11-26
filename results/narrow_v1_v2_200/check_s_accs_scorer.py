"""Independent check of the location score S in report.txt: recomputed with accs_scorer.spatial (the benchmark scorer)
and the sigmas of scorer_config.json, from results.csv. Run with .venv-ml (pandas):
    .venv-ml\\Scripts\\python.exe results\\narrow_v1_v2_200\\check_s_accs_scorer.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import pandas as pd  # noqa: E402
from accs_scorer import spatial  # noqa: E402

cfg = json.loads((REPO / "scorer_config.json").read_text(encoding="utf-8"))
d = pd.read_csv(HERE / "results.csv")
star = d[["center_x", "center_y"]].to_numpy(float)
sx, sy = cfg["sigma_x"] * cfg["spatial_k"], cfg["sigma_y"] * cfg["spatial_k"]
lines = [f"accs_scorer.spatial on {len(d)} clips, sigma_x {sx}, sigma_y {sy}"]
for name, xy in (("v1", d[["v1_x", "v1_y"]]), ("v2", d[["v2_x", "v2_y"]]),
                 ("centre", pd.DataFrame({"x": [0.5] * len(d), "y": [0.5] * len(d)}))):
    lines.append(f"  {name:7s} S {spatial(xy.to_numpy(float), star, sx, sy):.4f}")
for dn in ("day", "night"):
    m = d["day_night"] == dn
    for name, cols in (("v1", ["v1_x", "v1_y"]), ("v2", ["v2_x", "v2_y"])):
        lines.append(f"  {dn:5s} {name} S {spatial(d.loc[m, cols].to_numpy(float), star[m.to_numpy()], sx, sy):.4f}")
text = "\n".join(lines) + "\n"
(HERE / "check_s_accs_scorer.txt").write_text(text, encoding="utf-8")
print(text)
