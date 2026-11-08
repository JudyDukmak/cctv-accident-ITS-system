"""Check a run's recorded frame-cache manifest hash against the current cache (CPU only, no torch; shared by
analyse_predictions.py and, later, aggregate_runs.py).

A run records sha256(cache/frames224_v1/cache_manifest.json) as data_hashes.cache_manifest_sha256. A cache
rebuilt on another machine gets a different hash even when every frame is identical, because the manifest also
holds metadata of the build run. So:
  - recorded hash == current manifest hash: OK, nothing to report (behaviour as before);
  - otherwise the run's own manifest file must be supplied: its sha256 must equal the recorded hash, and every
    field except IGNORED (created_at, skipped_this_run, written_this_run) must equal the current manifest,
    including CORE (resize rule, JPEG quality, decoder, grid rule, model fps, windows hash, clip / frame / byte
    totals). Then a "documented deviation" note is returned; any other difference raises ValueError.
"""

import hashlib
import json
from pathlib import Path

IGNORED = ("created_at", "skipped_this_run", "written_this_run")  # metadata of the run that built the cache
CORE = ("resize_rule", "jpeg_quality", "decoder", "grid_rule", "model_fps", "windows_sha256", "clips_total",
        "clips_in_cache", "frames_this_run", "bytes_in_cache")  # must be present in both and equal


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_differences(current: dict, recorded: dict) -> list[str]:
    """Differences between two manifests in every field except IGNORED (missing CORE fields count too)."""
    diffs = [f"{k} missing in the {'current' if k not in current else 'run'} manifest"
             for k in CORE if k not in current or k not in recorded]
    for k in sorted(set(current) | set(recorded)):
        if k not in IGNORED and k in current and k in recorded and current[k] != recorded[k]:
            diffs.append(f"{k}: run {recorded[k]!r}, current {current[k]!r}")
        elif k not in IGNORED and k not in CORE and (k in current) != (k in recorded):
            diffs.append(f"{k} present only in the {'current' if k in current else 'run'} manifest")
    return diffs


def check_cache_manifest(recorded_sha: str, current_path: Path, run_manifest: Path | None) -> str | None:
    """None if the recorded hash equals the current manifest's; else the deviation note (see module doc).
    Raises ValueError if the hashes differ and run_manifest is missing, is not the recorded file, or differs
    from the current manifest outside IGNORED."""
    current_sha = sha256_file(current_path)
    if recorded_sha == current_sha:
        return None
    if run_manifest is None:
        raise ValueError(f"recorded cache manifest sha256 {recorded_sha[:12]}... != current {current_sha[:12]}...; "
                         "if the cache was rebuilt on another machine, pass that machine's cache_manifest.json "
                         "with --run-cache-manifest")
    run_manifest = Path(run_manifest)
    if not run_manifest.is_file():
        raise ValueError(f"--run-cache-manifest {run_manifest} not found")
    run_sha = sha256_file(run_manifest)
    if run_sha != recorded_sha:
        raise ValueError(f"{run_manifest} has sha256 {run_sha[:12]}..., not the run's recorded {recorded_sha[:12]}...")
    current = json.loads(Path(current_path).read_text(encoding="ascii"))
    recorded = json.loads(run_manifest.read_text(encoding="ascii"))
    diffs = core_differences(current, recorded)
    if diffs:
        raise ValueError("cache manifest of the run differs from the current one outside run metadata: "
                         + "; ".join(diffs))
    meta = [k for k in IGNORED if current.get(k) != recorded.get(k)]
    return (f"documented deviation: cache rebuilt on another machine, core fields identical (run manifest "
            f"{run_manifest} sha256 {run_sha[:12]}... = recorded; current {current_sha[:12]}...; differs only in "
            f"{', '.join(meta) or 'formatting'})")
