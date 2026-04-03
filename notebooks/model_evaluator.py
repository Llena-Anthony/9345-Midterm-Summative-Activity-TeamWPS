# # ======================================
# Programmer: Adrian Jay Soralde
# Date Programmed: April 02, 2026

# Objectives:
# Evaluation and comparison of all trained models (logistic_regression, naive_bayes, SVM)

# Tasks performed:
# Collect results from all models
# Compute evaluation metrics (Accuracy, Precision, Recall, F1-Score)
# Generate a confusion matrix
# Identify the best-performing model
# Save results to outputs/confusion_matrix.png
# Save results to outputs/model_comparison.csv
# # ======================================
from pathlib import Path
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



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
    }

# ==============================
# EXTRACT DECISION TREE METRICS
# ==============================
def extract_dt_metrics():
    df = pd.read_csv(DT_RESULTS)
    df.columns = df.columns.str.strip().str.lower()
    df["dataset_split"] = df["dataset_split"].str.strip().str.lower()

    def get_val(split, metric):
        row = df[df["dataset_split"] == split]
        if row.empty:
            print(f"WARNING: split '{split}' not found. Available: {df['dataset_split'].tolist()}")
            return 0.0
        if metric not in df.columns:
            print(f"WARNING: metric '{metric}' not found. Available: {df.columns.tolist()}")
            return 0.0
        return float(row[metric].values[0])

    return {
        "model": "Decision Tree",
        "accuracy": get_val("test", "accuracy"),
        "precision": get_val("test", "precision"),
        "recall": get_val("test", "recall"),
        "f1_score": get_val("test", "f1_score"),
    }

# ==============================
# EXTRACT RANDOM FOREST METRICS
# ==============================
def extract_rf_metrics():
    df = pd.read_csv(RF_RESULTS)
    df.columns = df.columns.str.strip().str.lower()
    df["dataset_split"] = df["dataset_split"].str.strip().str.lower()

    def get_val(split, metric):
        row = df[df["dataset_split"] == split]
        if row.empty:
            print(f"WARNING: split '{split}' not found. Available: {df['dataset_split'].tolist()}")
            return 0.0
        if metric not in df.columns:
            print(f"WARNING: metric '{metric}' not found. Available: {df.columns.tolist()}")
            return 0.0
        return float(row[metric].values[0])

    return {
        "model": "Random Forest",
        "accuracy": get_val("test", "accuracy"),
        "precision": get_val("test", "precision"),
        "recall": get_val("test", "recall"),
        "f1_score": get_val("test", "f1_score"),
    }

# ==============================
# LOAD DATA (for confusion matrix)
# ==============================
def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


# ==============================
# RECREATE MODELS (for CM only)
# ==============================
def get_predictions(model_name, X_train, X_test, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    if model_name == "Naive Bayes":
        model = GaussianNB()
        model.fit(X_res, y_res)
        return model.predict(X_test)

    elif model_name == "SVM":
        scaler = StandardScaler()
        X_res_scaled = scaler.fit_transform(X_res)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(probability=True, random_state=42)
        model.fit(X_res_scaled, y_res)
        return model.predict(X_test_scaled)

    elif model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(random_state=42)
        model.fit(X_res, y_res)
        return model.predict(X_test)

    elif model_name == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_res, y_res)
        return model.predict(X_test)

    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Check that it matches exactly.")


# ==============================
# SAVE CONFUSION MATRIX
# ==============================
def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"]
    )

    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

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

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Recreate predictions (ONLY for confusion matrix)
    y_pred = get_predictions(best_model, X_train, X_test, y_train)

    # Save confusion matrix
    save_confusion_matrix(y_test, y_pred, best_model)

    print("Outputs generated:")
    print("- model_comparison.csv")
    print("- confusion_matrix.png")

    print("NB:", nb)
    print("SVM:", svm)
    print("DT:", dt)
    print("RF:", rf)

if __name__ == "__main__":
    main()