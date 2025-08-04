from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_simulation_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    return pd.read_csv(path)


def is_chaotic(theta_series: np.ndarray, threshold_std: float = 1.5) -> bool:
    """
    A simple heuristic for chaos detection:
    If the standard deviation of the angle exceeds a threshold,
    label it as chaotic.
    """
    std_dev = np.std(theta_series)
    return std_dev > threshold_std


def label_data(df: pd.DataFrame, std_threshold: float = 1.5) -> pd.DataFrame:
    """
    Labels each simulation group (by drive_force_c) as chaotic or periodic.
    Adds a `label` column with 'chaotic' or 'periodic'.
    """
    if "drive_force_c" not in df.columns:
        raise ValueError("Missing 'drive_force_c' column for grouping")

    labeled_data = []
    for c_value, group in df.groupby("drive_force_c"):
        theta_series = group["theta"].values
        chaotic = is_chaotic(theta_series, threshold_std=std_threshold)
        group = group.copy()
        group["label"] = "chaotic" if chaotic else "periodic"
        labeled_data.append(group)

    return pd.concat(labeled_data, ignore_index=True)


def save_labeled_data(df: pd.DataFrame, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Labeled data saved to {path}")


def visualize_sample(df: pd.DataFrame, output_dir: Optional[str] = None):
    """
    Optional: Generates sample plots per label to visually inspect labeling.
    """
    output_dir = Path(output_dir or "data/plots_labeled")
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in df["label"].unique():
        subset = df[df["label"] == label]
        sample = subset.groupby("drive_force_c").head(1)
        for _, row in sample.iterrows():
            c_val = row["drive_force_c"]
            series = df[df["drive_force_c"] == c_val]
            plt.figure(figsize=(10, 4))
            plt.plot(series["t"], series["theta"])
            plt.title(f"Label: {label}, c={c_val:.3f}")
            plt.xlabel("Time")
            plt.ylabel("Theta (rad)")
            plt.grid(True)
            plt.tight_layout()
            fname = output_dir / f"{label}_c_{c_val:.3f}.png"
            plt.savefig(fname)
            plt.close()


if __name__ == "__main__":
    RAW_CSV_PATH = "data/raw/chaotic_pendulum_simulations.csv"
    LABELED_CSV_PATH = "data/processed/labeled_chaotic_pendulum.csv"

    try:
        df = load_simulation_data(RAW_CSV_PATH)
        labeled_df = label_data(df)
        save_labeled_data(labeled_df, LABELED_CSV_PATH)
        visualize_sample(labeled_df)
    except Exception as e:
        print(f"❌ Error during labeling: {e}")
