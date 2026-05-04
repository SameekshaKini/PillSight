# data_exploration.py
# filters all_labels.csv down to a clean subset and picks the top N classes
#
# the label column in the segmented images contains SHA hashes with exactly
# 2 images each (front + back), so we can't use it as a class directly.
# instead we group by label_code_id (the NDC manufacturer code) which gives
# 142-534 images per class in the top 10 -- enough to actually train on.
#
# outputs:
#   data/filtered_metadata.csv  -- the rows we'll use for training
#   data/class_map.json         -- {"54868": 0, "615": 1, ...}
#
# run this once before train.py

import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


def load_metadata(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError("CSV not found: %s\nCheck METADATA_CSV in config.py" % csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    print("[load] %d rows, columns: %s" % (len(df), df.columns.tolist()))
    return df


def check_nulls(df):
    missing = [c for c in cfg.REQUIRED_COLS_NULLCHECK if c not in df.columns]
    if missing:
        raise KeyError("Columns not found in CSV: %s" % missing)

    null_counts = df[cfg.REQUIRED_COLS_NULLCHECK].isnull().sum()
    if null_counts.sum() > 0:
        print("[nulls] WARNING -- missing values found:")
        print(null_counts[null_counts > 0].to_string())
    else:
        print("[nulls] no missing values in required columns")


def filter_reference_images(df):
    before = len(df)
    df = df[df["is_ref"] == True].reset_index(drop=True)
    print("[filter] is_ref=True: %d -> %d rows" % (before, len(df)))
    return df


def filter_segmented_subfolder(df):
    # only use the segmented images -- they have a clean white background
    # and are already 224x224, unlike the fcn_mix_weight ones
    before = len(df)
    df = df[df[cfg.COL_IMAGE_PATH].str.startswith(cfg.SUBFOLDER_FILTER)].reset_index(drop=True)
    print("[filter] segmented only: %d -> %d rows" % (before, len(df)))
    return df


def filter_existing_images(df):
    before = len(df)
    mask = df[cfg.COL_IMAGE_PATH].apply(lambda p: (cfg.IMAGE_DIR / p).exists())
    df = df[mask].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print("[images] WARNING: %d rows had missing image files" % dropped)
    else:
        print("[images] all %d image files found on disk" % len(df))
    return df


def select_top_classes(df, n):
    counts = (
        df.groupby(cfg.COL_LABEL)
          .size()
          .reset_index(name="image_count")
          .sort_values("image_count", ascending=False)
    )

    print("\n[classes] %d unique label_code_id groups, picking top %d\n" % (len(counts), n))
    print(counts.head(n).to_string(index=False))
    print()

    top_labels = counts.head(n)[cfg.COL_LABEL].tolist()
    filtered = df[df[cfg.COL_LABEL].isin(top_labels)].reset_index(drop=True).copy()

    # cast to string so it round-trips through JSON correctly
    # (JSON keys are always strings, and pandas reads int columns back as int64)
    filtered[cfg.COL_LABEL] = filtered[cfg.COL_LABEL].astype(str)

    print("[select] %d images across %d classes" % (len(filtered), n))
    return filtered


def build_class_map(df):
    classes = sorted(df[cfg.COL_LABEL].unique())
    return {cls: idx for idx, cls in enumerate(classes)}


def save_outputs(df, class_map):
    cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.FILTERED_METADATA_CSV, index=False)
    print("\n[save] filtered metadata -> %s" % cfg.FILTERED_METADATA_CSV)

    with open(cfg.CLASS_MAP_JSON, "w") as f:
        json.dump(class_map, f, indent=2)
    print("[save] class map -> %s" % cfg.CLASS_MAP_JSON)


def plot_class_distribution(df):
    counts = df.groupby(cfg.COL_LABEL).size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(counts)), counts.values, color="steelblue")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(["LC-%s" % lc for lc in counts.index], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("image count")
    ax.set_title("top %d classes by image count (label_code_id)" % cfg.NUM_CLASSES)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.OUTPUT_DIR / "class_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] class distribution -> %s" % out)


def print_summary(df, class_map):
    counts = df.groupby(cfg.COL_LABEL).size()
    total  = len(df)
    train  = int(total * cfg.TRAIN_RATIO)
    val    = int(total * cfg.VAL_RATIO)
    test   = total - train - val

    print("\n--- dataset summary ---")
    print("total images : %d" % total)
    print("classes      : %d" % len(class_map))
    print("min / max    : %d / %d" % (counts.min(), counts.max()))
    print("imbalance    : %.2fx" % (counts.max() / counts.min()))
    print("est split    : train=%d  val=%d  test=%d" % (train, val, test))
    print()
    print("class mapping:")
    for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
        print("  [%2d] label_code_id=%-8s  (%d images)" % (idx, name, counts[name]))

    if "is_front" in df.columns:
        print("\nfront/back split: front=%d  back=%d" % (df["is_front"].sum(), (~df["is_front"]).sum()))
    print()


def main():
    print("--- ePillID data exploration ---\n")

    df = load_metadata(cfg.METADATA_CSV)
    check_nulls(df)
    df = filter_reference_images(df)
    df = filter_segmented_subfolder(df)
    df = filter_existing_images(df)
    df = select_top_classes(df, cfg.NUM_CLASSES)

    class_map = build_class_map(df)
    save_outputs(df, class_map)
    plot_class_distribution(df)
    print_summary(df, class_map)

    print("done -- run train.py next")


if __name__ == "__main__":
    main()
