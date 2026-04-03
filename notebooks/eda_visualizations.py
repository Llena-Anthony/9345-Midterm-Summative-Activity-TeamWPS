from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
CLEANED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_stroke_data.csv"
OUTPUTS_DIR = BASE_DIR / "outputs" / "eda_visualizations"

KEY_FEATURES = ["age", "avg_glucose_level", "bmi"]


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw and cleaned stroke datasets."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEANED_DATA_PATH}")

    raw_df = pd.read_csv(RAW_DATA_PATH)
    cleaned_df = pd.read_csv(CLEANED_DATA_PATH)

    # Raw BMI contains string placeholders (e.g., 'N/A'), so coerce for plotting.
    raw_df["bmi"] = pd.to_numeric(raw_df["bmi"], errors="coerce")
    return raw_df, cleaned_df


def ensure_outputs_package() -> None:
    """Create outputs folder if it does not exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def create_correlation_heatmap(cleaned_df: pd.DataFrame) -> None:
    """Create and save a correlation heatmap from the cleaned dataset."""
    corr_matrix = cleaned_df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        annot=False,
        square=False,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap (Cleaned Stroke Dataset)")
    plt.tight_layout()

    output_path = OUTPUTS_DIR / "heatmap.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_histograms(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Create and save side-by-side histograms for key features."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)

    for idx, feature in enumerate(KEY_FEATURES):
        sns.histplot(raw_df[feature].dropna(), bins=30, kde=True, color="steelblue", ax=axes[0, idx])
        axes[0, idx].set_title(f"Raw: {feature}")
        axes[0, idx].set_xlabel(feature)

        sns.histplot(cleaned_df[feature].dropna(), bins=30, kde=True, color="darkorange", ax=axes[1, idx])
        axes[1, idx].set_title(f"Cleaned: {feature}")
        axes[1, idx].set_xlabel(feature)

    fig.suptitle("Key Feature Distributions (Raw vs Cleaned)", fontsize=14)
    plt.tight_layout()

    output_path = OUTPUTS_DIR / "histograms.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_boxplots(cleaned_df: pd.DataFrame) -> None:
    """Create and save boxplots for outlier inspection by stroke class."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

    for idx, feature in enumerate(KEY_FEATURES):
        sns.boxplot(
            data=cleaned_df,
            x="stroke",
            y=feature,
            hue="stroke",
            ax=axes[idx],
            palette="Set2",
            legend=False,
        )
        axes[idx].set_title(f"{feature} by stroke")
        axes[idx].set_xlabel("stroke (0=no, 1=yes)")

    fig.suptitle("Outlier Check with Boxplots (Cleaned Data)", fontsize=14)
    plt.tight_layout()

    output_path = OUTPUTS_DIR / "boxplots.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    ensure_outputs_package()
    raw_df, cleaned_df = load_datasets()

    create_correlation_heatmap(cleaned_df)
    create_histograms(raw_df, cleaned_df)
    create_boxplots(cleaned_df)

    print("Generated files:")
    print(f"- {OUTPUTS_DIR / 'heatmap.png'}")
    print(f"- {OUTPUTS_DIR / 'histograms.png'}")
    print(f"- {OUTPUTS_DIR / 'boxplots.png'}")


if __name__ == "__main__":
    main()

