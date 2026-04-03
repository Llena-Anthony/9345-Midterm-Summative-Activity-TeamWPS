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

# Outputs:
# outputs/naive_bayes_results.txt
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

NB_RESULTS = BASE_DIR / "outputs" / "naive_bayes" / "naive_bayes_results.txt"
SVM_RESULTS = BASE_DIR / "outputs" / "svm_results.txt"

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

if not NB_RESULTS.exists():
    raise FileNotFoundError(f"Naive Bayes results not found at {NB_RESULTS}")

if not SVM_RESULTS.exists():
    raise FileNotFoundError(f"SVM results not found at {SVM_RESULTS}")


# ==============================
# EXTRACT NAIVE BAYES METRICS
# ==============================
def extract_nb_metrics():
    with open(NB_RESULTS, "r") as f:
        text = f.read()

    # 🔹 Extract accuracy safely
    acc_match = re.search(r"(accuracy|Accuracy)[:\s]+([\d.]+)", text)
    accuracy = float(acc_match.group(2)) if acc_match else 0.0

    # 🔹 Extract weighted avg safely
    weighted_match = re.search(
        r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        text,
        re.IGNORECASE
    )

    if weighted_match:
        precision = float(weighted_match.group(1))
        recall = float(weighted_match.group(2))
        f1 = float(weighted_match.group(3))
    else:
        precision = recall = f1 = 0.0

    return {
        "model": "Naive Bayes",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# ==============================
# EXTRACT SVM METRICS
# ==============================
def extract_svm_metrics():
    with open(SVM_RESULTS, "r") as f:
        text = f.read()

    acc = float(re.search(r"Accuracy:\s+([\d.]+)", text).group(1))
    precision = float(re.search(r"Precision:\s+([\d.]+)", text).group(1))
    recall = float(re.search(r"Recall:\s+([\d.]+)", text).group(1))
    f1 = float(re.search(r"F1-score:\s+([\d.]+)", text).group(1))

    return {
        "model": "SVM",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
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
    if model_name == "Naive Bayes":
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        model = GaussianNB()
        model.fit(X_res, y_res)
        return model.predict(X_test)

    elif model_name == "SVM":
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_res_scaled = scaler.fit_transform(X_res)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(probability=True, random_state=42)
        model.fit(X_res_scaled, y_res)
        return model.predict(X_test_scaled)


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

    df = pd.DataFrame([nb, svm])

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

if __name__ == "__main__":
    main()