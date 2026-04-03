# # ======================================
# Programmer: Adrian Jay Soralde
# Date Programmed: April 02, 2026

# Objectives:
# Evaluation and comparison of all trained models (logistic_regression, naive_bayes, SVM)

# Tasks performed:
# Collect results from all models (idk how to)
# Compute evaluation metrics (Accuracy, Precision, Recall, F1-Score)
# Generate a confusion matrix
# Identify the best-performing model
# Save results to outputs/confusion_matrix.png
# Save results to outputs/model_comparison.csv

# Outputs:
# outputs/naive_bayes_results.txt
# # ======================================
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEANED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_stroke_data.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"

def ensure_output_dir():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load and preprocess dataset."""
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEANED_DATA_PATH}")

    df = pd.read_csv(CLEANED_DATA_PATH)

    df = pd.get_dummies(df, drop_first=True)

    return df


def prepare_data(df: pd.DataFrame):
    """Split dataset."""
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def compare_models():
    nb = pd.read_csv(OUTPUTS_DIR / "naive_bayes_results.txt")
    svm = pd.read_csv(OUTPUTS_DIR / "svm_results.txt")

    df = pd.concat([nb,svm], ignore_index=True)

    df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    best = df.loc[df["f1_score"].idxmax()]
    print("Best Model:", best["model"])


def save_confusion_matrix(model, X_test, y_test, model_name: str):
    """Generate confusion matrix plot."""
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")

    plt.title(f"Confusion Matrix ({model_name})")

    output_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_models():
    """Main evaluation logic."""
    ensure_output_dir()

    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    models = compare_models()

    results = []
    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

    results_df = pd.DataFrame(results)
    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"Best Model: {best_name} (F1-score: {best_score:.4f})")

    save_confusion_matrix(best_model, X_test, y_test, best_name)


def main():
    evaluate_models()

    print("Generated files:")
    print(f"- {OUTPUTS_DIR / 'model_comparison.csv'}")
    print(f"- {OUTPUTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()