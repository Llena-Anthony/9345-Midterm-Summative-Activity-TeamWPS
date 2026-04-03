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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
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
    # 1. Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 2. Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train model
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )

    model.fit(X_train_scaled, y_train_resampled)

    # 4. Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    return model, scaler, y_pred, y_proba

# ======================================
# SAVE RESULTS TO FILE
# ======================================

def save_results_to_file(
    acc_test, roc_test, report_test, cm_test, precision_test, recall_test, f1_test, specificity_test,
    acc_unseen, roc_unseen, report_unseen, cm_unseen, precision_unseen, recall_unseen, f1_unseen, specificity_unseen
):
    with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SVM STROKE PREDICTION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        # =========================
        # TEST SET
        # =========================
        f.write("TEST SET RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {acc_test:.4f}\n")
        f.write(f"Precision: {precision_test:.4f}\n")
        f.write(f"Recall: {recall_test:.4f}\n")
        f.write(f"F1-score: {f1_test:.4f}\n")
        f.write(f"Specificity: {specificity_test:.4f}\n")
        f.write(f"ROC-AUC Score: {roc_test:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(report_test + "\n")

        f.write("Confusion Matrix:\n")
        f.write(f"{cm_test}\n\n")

        # =========================
        # UNSEEN SET
        # =========================
        f.write("UNSEEN SET RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {acc_unseen:.4f}\n")
        f.write(f"Precision: {precision_unseen:.4f}\n")
        f.write(f"Recall: {recall_unseen:.4f}\n")
        f.write(f"F1-score: {f1_unseen:.4f}\n")
        f.write(f"Specificity: {specificity_unseen:.4f}\n")
        f.write(f"ROC-AUC Score: {roc_unseen:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(report_unseen + "\n")

        f.write("Confusion Matrix:\n")
        f.write(f"{cm_unseen}\n\n")

        f.write("=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")

    print(" Results saved (Test + Unseen)")

# ======================================
# ROC CURVE PLOT
# ======================================

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


# ======================================
# CONFUSION MATRIX PLOT
# ======================================

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

# ======================================
# TEN-FOLD CROSS VALIDATION
# ======================================

def cross_validate_svm(model, X, y, folds=10):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': [],
        'roc_auc': []
    }

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_val, y_pred))
        metrics['f1'].append(f1_score(y_val, y_pred))
        metrics['specificity'].append(tn / (tn + fp))
        metrics['roc_auc'].append(roc_auc_score(y_val, y_proba))

    # Average metrics across folds
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics


# ======================================
# MAIN EXECUTION
# ======================================
def main():
    # 1. Train the model on training data (with SMOTE)
    model, scaler, y_test_pred, y_test_proba = train_model()

    # ====================================================
    # TEST SET EVALUATION
    # ====================================================
    acc_test = accuracy_score(y_test, y_test_pred)
    roc_test = roc_auc_score(y_test, y_test_proba)
    cm_test = confusion_matrix(y_test, y_test_pred)
    report_test = classification_report(y_test, y_test_pred)

    tn, fp, fn, tp = cm_test.ravel()
    precision_test = precision_score(y_test, y_test_pred, zero_division=0)
    recall_test = recall_score(y_test, y_test_pred, zero_division=0)
    f1_test = f1_score(y_test, y_test_pred, zero_division=0)
    specificity_test = tn / (tn + fp)

    print("SVM Test Set Results")
    print("===================")
    print(f"Accuracy: {acc_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall: {recall_test:.4f}")
    print(f"F1-score: {f1_test:.4f}")
    print(f"Specificity: {specificity_test:.4f}")
    print(f"ROC-AUC: {roc_test:.4f}")
    print("\nConfusion Matrix:")
    print(cm_test)
    print(report_test)

    plot_roc_curve(y_test, y_test_proba)
    plot_confusion_matrix(cm_test)

    # ====================================================
    # UNSEEN SET EVALUATION
    # ====================================================
    X_unseen_scaled = scaler.transform(X_unseen)
    y_unseen_pred = model.predict(X_unseen_scaled)
    y_unseen_proba = model.predict_proba(X_unseen_scaled)[:, 1]

    acc_unseen = accuracy_score(y_unseen, y_unseen_pred)
    roc_unseen = roc_auc_score(y_unseen, y_unseen_proba)
    cm_unseen = confusion_matrix(y_unseen, y_unseen_pred)
    report_unseen = classification_report(y_unseen, y_unseen_pred)

    tn, fp, fn, tp = cm_unseen.ravel()
    precision_unseen = tp / (tp + fp)
    recall_unseen = tp / (tp + fn)
    f1_unseen = 2 * (precision_unseen * recall_unseen) / (precision_unseen + recall_unseen)
    specificity_unseen = tn / (tn + fp)

    print("\nSVM Unseen Set Results")
    print("=====================")
    print(f"Accuracy: {acc_unseen:.4f}")
    print(f"Precision: {precision_unseen:.4f}")
    print(f"Recall: {recall_unseen:.4f}")
    print(f"F1-score: {f1_unseen:.4f}")
    print(f"Specificity: {specificity_unseen:.4f}")
    print(f"ROC-AUC: {roc_unseen:.4f}")
    print("\nConfusion Matrix:")
    print(cm_unseen)
    print(report_unseen)

    save_results_to_file(
        acc_test, roc_test, report_test, cm_test,
        precision_test, recall_test, f1_test, specificity_test,
        acc_unseen, roc_unseen, report_unseen, cm_unseen,
        precision_unseen, recall_unseen, f1_unseen, specificity_unseen
    )
# ======================================
# RUN THE SCRIPT
# ======================================

if __name__ == "__main__":
    main()