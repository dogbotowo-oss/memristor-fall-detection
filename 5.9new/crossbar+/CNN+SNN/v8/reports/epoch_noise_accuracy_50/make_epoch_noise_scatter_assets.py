from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\epoch_noise_accuracy_50")
DATA_DIR = ROOT / "shuju"
IMAGE_DIR = ROOT / "image"

SOURCE_FILES = {
    "Gaussian": DATA_DIR / "Gaussian_epoch_accuracy.csv",
    "Poisson": DATA_DIR / "Poisson_epoch_accuracy.csv",
    "SaltPepper": DATA_DIR / "SaltPepper_epoch_accuracy.csv",
}

OUTPUT_LONG_FILES = {
    "Gaussian": DATA_DIR / "Gaussian_epoch_accuracy_scatter_long.csv",
    "Poisson": DATA_DIR / "Poisson_epoch_accuracy_scatter_long.csv",
    "SaltPepper": DATA_DIR / "SaltPepper_epoch_accuracy_scatter_long.csv",
}

PLOT_FILES = {
    "Gaussian": IMAGE_DIR / "Gaussian_epoch_accuracy_scatter_cloud.png",
    "Poisson": IMAGE_DIR / "Poisson_epoch_accuracy_scatter_cloud.png",
    "SaltPepper": IMAGE_DIR / "SaltPepper_epoch_accuracy_scatter_cloud.png",
}

COMBINED_LONG_PATH = DATA_DIR / "All_noise_epoch_accuracy_scatter_ready.csv"
PANEL_PATH = IMAGE_DIR / "epoch_noise_scatter_cloud_panel.png"

NOISE_ORDER = [0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5]
PALETTES = {
    "Gaussian": ["#DCEAF7", "#C7DBF0", "#AFC8E4", "#8FB1D4", "#6C93BD", "#4D739B", "#35567C", "#1F3E5A"],
    "Poisson": ["#F4E4D1", "#EDD2B5", "#E3BB94", "#D89F70", "#C77E4D", "#AB6337", "#8B4C2A", "#6B3921"],
    "SaltPepper": ["#F3E1E7", "#EBCFD8", "#DDB3C1", "#CF95AA", "#BE788E", "#A85F75", "#89485C", "#683441"],
}


def parse_noise_percent(column_name: str) -> float:
    marker = column_name.replace("Noise_", "").replace("_Test", "").replace("%", "")
    return float(marker)


def wide_to_long(noise_type: str, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    value_columns = [column for column in df.columns if column != "Epoch"]
    long_df = df.melt(id_vars=["Epoch"], value_vars=value_columns, var_name="NoiseLabel", value_name="AccuracyPercent")
    long_df["NoisePercent"] = long_df["NoiseLabel"].map(parse_noise_percent)
    long_df["NoiseLevel"] = long_df["NoisePercent"] / 100.0
    long_df["NoiseType"] = noise_type
    long_df["NoiseRank"] = long_df["NoisePercent"].map({value: index for index, value in enumerate(NOISE_ORDER)})
    long_df = long_df[["Epoch", "NoiseType", "NoiseLabel", "NoiseLevel", "NoisePercent", "NoiseRank", "AccuracyPercent"]]
    return long_df.sort_values(["NoisePercent", "Epoch"]).reset_index(drop=True)


def plot_single(ax: plt.Axes, df: pd.DataFrame, noise_type: str) -> None:
    palette = PALETTES[noise_type]
    for color, noise_percent in zip(palette, NOISE_ORDER):
        subset = df[df["NoisePercent"] == noise_percent]
        ax.scatter(
            subset["Epoch"],
            subset["AccuracyPercent"],
            s=18,
            alpha=0.82,
            color=color,
            edgecolors="none",
            label=f"{noise_percent:g}%",
        )
    ax.set_title(noise_type, fontsize=13, color="#243447", pad=10)
    ax.set_xlabel("Epoch", fontsize=11, color="#243447")
    ax.set_ylabel("Accuracy (%)", fontsize=11, color="#243447")
    ax.set_xlim(1, max(int(df["Epoch"].max()), 50))
    ax.set_ylim(30, 80)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.28, color="#8FA1B3")
    ax.tick_params(labelsize=9, colors="#3C4D5D")
    for spine in ax.spines.values():
        spine.set_color("#C9D3DD")
        spine.set_linewidth(0.9)


def save_single_figure(df: pd.DataFrame, noise_type: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")
    plot_single(ax, df, noise_type)
    ax.legend(
        title="Noise Level",
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.tight_layout()
    fig.savefig(PLOT_FILES[noise_type], bbox_inches="tight")
    plt.close(fig)


def save_panel_figure(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7), dpi=220, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, noise_type in zip(axes, ["Gaussian", "Poisson", "SaltPepper"]):
        ax.set_facecolor("#FBFCFE")
        plot_single(ax, frames[noise_type], noise_type)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Noise Level",
        ncol=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PANEL_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    frame_map: dict[str, pd.DataFrame] = {}
    for noise_type, source_file in SOURCE_FILES.items():
        long_df = wide_to_long(noise_type, source_file)
        long_df.to_csv(OUTPUT_LONG_FILES[noise_type], index=False, encoding="utf-8-sig")
        save_single_figure(long_df, noise_type)
        frame_map[noise_type] = long_df
        all_frames.append(long_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(COMBINED_LONG_PATH, index=False, encoding="utf-8-sig")
    save_panel_figure(frame_map)

    print(OUTPUT_LONG_FILES["Gaussian"])
    print(OUTPUT_LONG_FILES["Poisson"])
    print(OUTPUT_LONG_FILES["SaltPepper"])
    print(COMBINED_LONG_PATH)
    print(PLOT_FILES["Gaussian"])
    print(PLOT_FILES["Poisson"])
    print(PLOT_FILES["SaltPepper"])
    print(PANEL_PATH)


if __name__ == "__main__":
    main()
