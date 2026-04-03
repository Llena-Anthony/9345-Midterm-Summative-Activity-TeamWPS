# # ======================================
# Programmer: Leo Delos Reyes
# Date Programmed: April 02, 2026

# Objectives:
# Model 3 – Support Vector Machine (SVM)
# Responsible for implementing an advanced classification model.

# Tasks performed:
# Load the processed dataset
# Train a Support Vector Machine model
# Perform basic parameter tuning
# Evaluate model performance

# Outputs:
# outputs/svm_results.txt
# # ======================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================
# PATHS - Define the path to your data file
# ======================================

# Get the current script's directory
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent

# Path to your processed data file
DATA_PATH = SCRIPT_DIR / "data" / "processed"

# Load splits
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")
X_unseen = pd.read_csv(DATA_PATH / "X_unseen.csv")

y_train = pd.read_csv(DATA_PATH / "y_train.csv").values.ravel()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").values.ravel()
y_unseen = pd.read_csv(DATA_PATH / "y_unseen.csv").values.ravel()

# Path to store the result
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "svm_results.txt"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

ROC_IMG_PATH = SCRIPT_DIR / "outputs" / "roc_curve_SVM.png"
CM_IMG_PATH = SCRIPT_DIR / "outputs" / "confusion_matrix_SVM.png"


# ======================================
# TRAIN MODEL
# ======================================

def train_model():
    print("\nTraining SVM using provided split...")

    # 1. Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 2. Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)  # ✅ use resampled data
    X_test_scaled = scaler.transform(X_test)

    # 3. Train model
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )

    model.fit(X_train_scaled, y_train_resampled)  # ✅ fit with resampled labels

    # 4. Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    return model, scaler, y_pred, y_proba

# ======================================
# SAVE RESULTS TO FILE
# ======================================

def save_results_to_file(acc, roc, report, cm, precision, recall, f1, specificity):
    with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SVM STROKE PREDICTION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"ROC-AUC Score: {roc:.4f}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(report + "\n")


        f.write("CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write(f"Matrix:\n{cm}\n\n")

        f.write("=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")

    print("Results saved")


# ============================================================================
# ROC CURVE PLOT
# ============================================================================

def plot_roc_curve(y_test, y_pred_proba):
    """Plot and save ROC curve"""

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Ensure directory exists
    ROC_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_score:.3f})", color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Stroke Prediction (SVM)", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save image
    plt.savefig(ROC_IMG_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" ROC curve saved")


# ============================================================================
# CONFUSION MATRIX PLOT
# ============================================================================

def plot_confusion_matrix(cm):
    """Plot and save confusion matrix"""

    # Ensure directory exists
    CM_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    plt.title('Confusion Matrix - SVM Stroke Prediction', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Save image
    plt.savefig(CM_IMG_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Confusion matrix saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    model, scaler, y_pred, y_proba = train_model()

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    specificity = tn / (tn + fp)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    save_results_to_file(acc, roc, report, cm, precision, recall, f1, specificity)

    plot_roc_curve(y_test, y_proba)
    plot_confusion_matrix(cm)

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()