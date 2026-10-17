
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "heuristic"
OUTPUT = RESULTS / "evaluation_results"
GROUND_TRUTH = ROOT / "real_test_set.csv"

PREDICTION_FILES = {
    "Naive": RESULTS / "output_naive-400.csv",
    "Optical Flow": RESULTS / "output_optical_flow.csv",
    "Bounding-Box Dynamics": RESULTS / "output_bbox_dynamics_test.csv",
    "Ensemble": ROOT / "ACCIDENT" / "baselines" / "heuristic" / "output_ensemble_test.csv",
}

REQUIRED_PREDICTION_COLUMNS = [
    "path",
    "accident_time",
    "center_x",
    "center_y",
]


def normalize_path(value):
    value = str(value).strip().replace("\\", "/")
    return value.rsplit("/", 1)[-1]


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if "path" not in df.columns:
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed:
            df = df.rename(columns={unnamed[0]: "path"})
        else:
            raise ValueError(f"No path column found in {path}")

    df["video_key"] = df["path"].map(normalize_path)

    if df["video_key"].duplicated().any():
        duplicates = df.loc[
            df["video_key"].duplicated(keep=False), "video_key"
        ].unique()[:10]
        raise ValueError(
            f"Duplicate video paths in {path}: {list(duplicates)}"
        )

    return df


def prepare_predictions(method, path):
    df = load_csv(path)

    missing = [
        column
        for column in REQUIRED_PREDICTION_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"{method} is missing columns: {missing}")

    df = df[
        ["video_key", "accident_time", "center_x", "center_y"]
    ].copy()

    df = df.rename(
        columns={
            "accident_time": f"{method}_time",
            "center_x": f"{method}_x",
            "center_y": f"{method}_y",
        }
    )

    for column in df.columns:
        if column != "video_key":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def temporal_accuracy(predictions, truth, sigma):
    errors = predictions - truth
    return float(np.mean(np.exp(-(errors ** 2) / (2 * sigma ** 2))))


def spatial_accuracy(pred_x, pred_y, true_x, true_y, sigma_x, sigma_y):
    dx = pred_x - true_x
    dy = pred_y - true_y

    scores = np.exp(
        -0.5 * (
            (dx / sigma_x) ** 2
            + (dy / sigma_y) ** 2
        )
    )
    return float(np.mean(scores))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    gt = load_csv(GROUND_TRUTH)

    required_gt = [
        "accident_time",
        "center_x",
        "center_y",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    missing_gt = [c for c in required_gt if c not in gt.columns]
    if missing_gt:
        raise ValueError(f"Ground truth is missing columns: {missing_gt}")

    gt = gt[
        ["video_key", "accident_time", "center_x", "center_y",
         "x1", "y1", "x2", "y2"]
    ].copy()

    gt = gt.rename(
        columns={
            "accident_time": "true_time",
            "center_x": "true_x",
            "center_y": "true_y",
        }
    )

    for column in gt.columns:
        if column != "video_key":
            gt[column] = pd.to_numeric(gt[column], errors="coerce")

    gt["bbox_width"] = gt["x2"] - gt["x1"]
    gt["bbox_height"] = gt["y2"] - gt["y1"]

    predictions = {
        method: prepare_predictions(method, path)
        for method, path in PREDICTION_FILES.items()
    }

    common_keys = set(gt["video_key"])
    counts = {
        "Ground Truth": len(gt),
    }

    for method, df in predictions.items():
        counts[method] = len(df)
        common_keys &= set(df["video_key"])

    if not common_keys:
        raise ValueError(
            "No videos match across ground truth and all four methods. "
            "Check the path values in the CSV files."
        )

    common_keys = sorted(common_keys)

    matched = gt[gt["video_key"].isin(common_keys)].copy()

    for method, df in predictions.items():
        matched = matched.merge(
            df[df["video_key"].isin(common_keys)],
            on="video_key",
            how="inner",
            validate="one_to_one",
        )

    numeric_columns = [
        c for c in matched.columns if c != "video_key"
    ]
    valid_mask = np.isfinite(matched[numeric_columns]).all(axis=1)
    valid_mask &= matched["bbox_width"] > 0
    valid_mask &= matched["bbox_height"] > 0

    invalid_count = int((~valid_mask).sum())
    matched = matched.loc[valid_mask].copy()

    if matched.empty:
        raise ValueError("No valid matched videos remain after data checks.")

    sigma_x_base = float(matched["bbox_width"].mean())
    sigma_y_base = float(matched["bbox_height"].mean())

    if sigma_x_base <= 0 or sigma_y_base <= 0:
        raise ValueError("Ground-truth bounding-box dimensions are invalid.")

    metrics = []
    per_video = matched[["video_key", "true_time", "true_x", "true_y"]].copy()

    for method in PREDICTION_FILES:
        pred_time = matched[f"{method}_time"].to_numpy(dtype=float)
        pred_x = matched[f"{method}_x"].to_numpy(dtype=float)
        pred_y = matched[f"{method}_y"].to_numpy(dtype=float)

        true_time = matched["true_time"].to_numpy(dtype=float)
        true_x = matched["true_x"].to_numpy(dtype=float)
        true_y = matched["true_y"].to_numpy(dtype=float)

        time_error = pred_time - true_time
        center_error = np.sqrt(
            (pred_x - true_x) ** 2 + (pred_y - true_y) ** 2
        )

        row = {
            "Method": method,
            "Matched Videos": len(matched),
            "Temporal MAE (seconds)": float(np.mean(np.abs(time_error))),
            "Temporal RMSE (seconds)": float(np.sqrt(np.mean(time_error ** 2))),
            "Mean Center Error (normalized units)": float(np.mean(center_error)),
        }

        for sigma in (0.5, 1, 2):
            row[f"Temporal Accuracy sigma={sigma}"] = temporal_accuracy(
                pred_time, true_time, sigma
            )
            row[f"Spatial Accuracy sigma={sigma}"] = spatial_accuracy(
                pred_x,
                pred_y,
                true_x,
                true_y,
                sigma_x_base * sigma,
                sigma_y_base * sigma,
            )

        metrics.append(row)

        per_video[f"{method}_time_error_seconds"] = time_error
        per_video[f"{method}_center_error"] = center_error

        for sigma in (0.5, 1, 2):
            per_video[f"{method}_temporal_accuracy_sigma_{sigma}"] = np.exp(
                -(time_error ** 2) / (2 * sigma ** 2)
            )

            dx = pred_x - true_x
            dy = pred_y - true_y
            per_video[f"{method}_spatial_accuracy_sigma_{sigma}"] = np.exp(
                -0.5 * (
                    (dx / (sigma_x_base * sigma)) ** 2
                    + (dy / (sigma_y_base * sigma)) ** 2
                )
            )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT / "metrics_comparison.csv", index=False)
    per_video.to_csv(OUTPUT / "per_video_metrics.csv", index=False)

    gt_keys = set(gt["video_key"])
    report = [
        "HEURISTIC EVALUATION REPORT",
        "=" * 60,
        f"Ground-truth videos: {len(gt_keys)}",
        f"Videos shared by ground truth and all four methods: {len(common_keys)}",
        f"Valid videos used for metrics: {len(matched)}",
        f"Rows excluded due to missing/invalid numeric values: {invalid_count}",
        "",
        "Input prediction file counts:",
    ]

    for method, count in counts.items():
        report.append(f"{method}: {count}")

    report.extend([
        "",
        f"Mean ground-truth bounding-box width: {sigma_x_base:.8f}",
        f"Mean ground-truth bounding-box height: {sigma_y_base:.8f}",
        "",
        "Metric definitions:",
        "- Temporal accuracy: mean exp(-(prediction error^2)/(2*sigma^2)).",
        "- Spatial accuracy: Gaussian score of normalized center errors.",
        "- Spatial sigma uses mean ground-truth box width/height multiplied by sigma.",
        "- All four methods are evaluated on the same valid matched videos.",
        "",
        "Ground-truth videos missing from the common intersection:",
        *sorted(gt_keys - set(common_keys)),
    ])

    for method, df in predictions.items():
        missing = gt_keys - set(df["video_key"])
        report.extend([
            "",
            f"Ground-truth videos without {method} predictions: {len(missing)}",
            *sorted(missing),
        ])

    (OUTPUT / "matching_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nEVALUATION COMPLETED")
    print(f"Valid matched videos: {len(matched)}")
    print(f"Results folder: {OUTPUT}")
    print("\nMetric comparison:")
    print(metrics_df.to_string(index=False))
    print("\nFiles saved:")
    print(OUTPUT / "metrics_comparison.csv")
    print(OUTPUT / "per_video_metrics.csv")
    print(OUTPUT / "matching_report.txt")


if __name__ == "__main__":
    main()
