# # ======================================
# Programmer: Adrian Jay Soralde
# Date Programmed: April 02, 2026

# Objectives:
# Evaluation and comparison of all trained models (naive_bayes, SVM, decision tree, random_forest)

# Tasks performed:
# Collect results from all models
# Compute evaluation metrics (Accuracy, Precision, Recall, F1-Score)
# Identify the best-performing model
# Save results to outputs/model_comparison.csv
# # ======================================
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

NB_RESULTS = BASE_DIR / "outputs" / "naive_bayes" / "naive_bayes_results.csv"
SVM_RESULTS = BASE_DIR / "outputs" /"support_vector_machine"/ "svm_results.csv"
DT_RESULTS = BASE_DIR / "outputs" / "decision_tree" / "decision_tree_metrics.csv"
RF_RESULTS = BASE_DIR / "outputs" / "random_forest" / "random_forest_metrics.csv"

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

if not NB_RESULTS.exists():
    raise FileNotFoundError(f"Naive Bayes results not found at {NB_RESULTS}")

if not SVM_RESULTS.exists():
    raise FileNotFoundError(f"SVM results not found at {SVM_RESULTS}")

if not DT_RESULTS.exists():
    raise FileNotFoundError(f"Decision Tree results not found at {DT_RESULTS}")

if not RF_RESULTS.exists():
    raise FileNotFoundError(f"Random Forest results not found at {RF_RESULTS}")

# ==============================
# EXTRACT NAIVE BAYES METRICS
# ==============================
def extract_nb_metrics():
    df = pd.read_csv(NB_RESULTS)
    print("NB Columns:", df.columns.tolist())
    df.columns = df.columns.str.strip()
    print("NB Columns after strip:", df.columns.tolist())
    df["Metric"] = df["Metric"].str.strip().str.lower()

    def get_val(metric):
        row = df[df["Metric"] == metric]

        if row.empty:
            print(f"WARNING: metric '{metric}' not found. Available: {df['Metric'].tolist()}")
            return 0.0
        return float(row["Test_Set"].values[0])
    return {
        "model": "Naive Bayes",
        "accuracy": get_val("accuracy"),
        "precision": get_val("precision"),
        "recall": get_val("recall"),
        "f1_score": get_val("f1-score"),
        "specificity": get_val("specificity"),
        "roc_auc": get_val("roc-auc"),
    }


# ==============================
# EXTRACT SVM METRICS
# ==============================
def extract_svm_metrics():
    df = pd.read_csv(SVM_RESULTS)
    df.columns = df.columns.str.strip()
    df["Metric"] = df["Metric"].str.strip().str.lower()

    def get_val(metric):
        row = df[df["Metric"] == metric]
        return float(row["Test_Set"].values[0]) if not row.empty else 0.0

    return {
        "model": "SVM",
        "accuracy": get_val("accuracy"),
        "precision": get_val("precision"),
        "recall": get_val("recall"),
        "f1_score": get_val("f1-score"),
        "specificity": get_val("specificity"),
        "roc_auc": get_val("roc-auc"),
    }

# ==============================
# EXTRACT DECISION TREE METRICS
# ==============================
def extract_dt_metrics():
    df = pd.read_csv(DT_RESULTS)
    df.columns = df.columns.str.strip()
    df["Metric"] = df["Metric"].str.strip().str.lower()

    def get_val(metric):
        row = df[df["Metric"] == metric]
        if row.empty:
            print(f"WARNING: metric '{metric}' not found. Available: {df['Metric'].tolist()}")
            return 0.0
        return float(row["Test Set"].values[0])

    return {
        "model": "Decision Tree",
        "accuracy": get_val("accuracy"),
        "precision": get_val("precision"),
        "recall": get_val("recall"),
        "f1_score": get_val("f1-score"),
        "specificity": get_val("specificity"),
        "roc_auc": get_val("roc-auc"),
    }

# ==============================
# EXTRACT RANDOM FOREST METRICS
# ==============================
def extract_rf_metrics():
    df = pd.read_csv(RF_RESULTS)
    df.columns = df.columns.str.strip()
    df["Metric"] = df["Metric"].str.strip().str.lower()

    def get_val(metric):
        row = df[df["Metric"] == metric]
        if row.empty:
            print(f"WARNING: metric '{metric}' not found. Available: {df['Metric'].tolist()}")
            return 0.0
        return float(row["Test Set"].values[0])

    return {
        "model": "Random Forest",
        "accuracy": get_val("accuracy"),
        "precision": get_val("precision"),
        "recall": get_val("recall"),
        "f1_score": get_val("f1-score"),
        "specificity": get_val("specificity"),
        "roc_auc": get_val("roc-auc"),
    }



# ==============================
# MAIN PIPELINE
# ==============================
def main():
    nb = extract_nb_metrics()
    svm = extract_svm_metrics()
    dt = extract_dt_metrics()
    rf = extract_rf_metrics()

    df = pd.DataFrame([nb, svm, dt, rf])

    # Save comparison
    df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    # Select best model (F1-score)
    best = df.loc[df["f1_score"].idxmax()]
    best_model = best["model"]

    print("Best Model:", best_model)

    print("Outputs generated:")
    print("- model_comparison.csv")

    print("NB:", nb)
    print("SVM:", svm)
    print("DT:", dt)
    print("RF:", rf)

if __name__ == "__main__":
    main()