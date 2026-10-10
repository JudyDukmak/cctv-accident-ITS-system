import sys, os
import pandas as pd

csv = sys.argv[1] if len(sys.argv) > 1 else "Dataset/metadata-real.csv"
root = sys.argv[2] if len(sys.argv) > 2 else None   # folder that contains real_videos/
df = pd.read_csv(csv)

print("rows:", len(df))
label_cols = ["type", "accident_time", "accident_frame",
              "center_x", "center_y", "x1", "y1", "x2", "y2"]
print("\nmissing values per label column:\n", df[label_cols].isna().sum())

print("\ncollision types:\n", df["type"].value_counts(dropna=False))
print("\nsplit_in_distribution:\n", df["split_in_distribution"].value_counts())
print("\nsplit_geo_aware:\n", df["split_geo_aware"].value_counts())
for c in ["quality", "day_time", "weather", "scene_layout", "region", "rollover"]:
    print(f"\n{c}:\n", df[c].value_counts(dropna=False))

# sanity checks
bad_time = df[(df.accident_time < 0) | (df.accident_time > df.duration)]
print("\naccident_time outside [0, duration]:", len(bad_time))
coords = ["center_x", "center_y", "x1", "y1", "x2", "y2"]
print("coords outside [0,1]:", int(((df[coords] < 0) | (df[coords] > 1)).any(axis=1).sum()))
inside = (df.center_x.between(df.x1, df.x2)) & (df.center_y.between(df.y1, df.y2))
print("centre not inside bbox:", int((~inside).sum()))

df["fps_from_duration"] = df.no_frames / df.duration
df["fps_from_accident"] = df.accident_frame / df.accident_time
df["fps_gap"] = (df.fps_from_duration - df.fps_from_accident).abs()
print("\nfps summary:\n", df[["fps_from_duration", "fps_from_accident", "fps_gap"]].describe())
print("clips where the two fps estimates differ by >1:", int((df.fps_gap > 1).sum()))
print("frames before accident (min/median):",
      df.accident_frame.min(), df.accident_frame.median())

df["source_video"] = (df.path.str.replace(r"_\d+\.mp4$", "", regex=True)
                      .str.split("/").str[-1])
print("\nclips per source video (max):", df.source_video.value_counts().max(),
      "| distinct source videos:", df.source_video.nunique())

if root:
    missing = [p for p in df.path if not os.path.exists(os.path.join(root, p))]
    print("\nmissing video files:", len(missing), missing[:5])