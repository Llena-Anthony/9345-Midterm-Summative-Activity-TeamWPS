# ======================================
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
# outputs/svm_results.csv
# ======================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, roc_curve, precision_score, recall_score,
                             f1_score)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================
# PATHS
# ======================================
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = SCRIPT_DIR / "data" / "processed"

# Load splits
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")
X_unseen = pd.read_csv(DATA_PATH / "X_unseen.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv").values.ravel()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").values.ravel()
y_unseen = pd.read_csv(DATA_PATH / "y_unseen.csv").values.ravel()

# Output directory
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "support_vector_machine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "svm_results.csv"


# ======================================
# PROPER CROSS-VALIDATION WITH CORRECT PIPELINE ORDER
# ======================================
def cross_validate_svm(X, y, folds=10):
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
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale FIRST
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        # THEN apply SMOTE on scaled training data
        smote = SMOTE(random_state=42)
        X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr_scaled, y_tr)

        # Train model
        model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        model.fit(X_tr_resampled, y_tr_resampled)

        # Predictions
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]

        # Calculate metrics
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        metrics['roc_auc'].append(roc_auc_score(y_val, y_proba))

    return {k: np.mean(v) for k, v in metrics.items()}


# ======================================
# TRAIN MODEL WITH CORRECT PIPELINE ORDER
# ======================================
def train_model():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Train model
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )

    model.fit(X_train_resampled, y_train_resampled)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    return model, scaler, y_pred, y_proba, y_train_resampled


# ======================================
# EVALUATION METRICS
# ======================================
def evaluate_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Handle division by zero
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "roc_auc": roc_auc_score(y_true, y_proba)
    }


# ======================================
# PLOTTING FUNCTIONS
# ======================================
def plot_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    plt.title(f'Confusion Matrix - SVM Stroke Prediction ({dataset_name})', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Save image
    plt.savefig(OUTPUT_DIR / f'confusion_matrix_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_proba, dataset_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_score:.3f})", color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f'ROC Curve - Stroke Prediction SVM ({dataset_name})', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save image
    plt.savefig(OUTPUT_DIR / f'roc_curve_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


# ======================================
# SAVE RESULTS
# ======================================
def save_results_to_file(test_results, unseen_results, cv_results):

    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC'],
        'Cross-Validation': [
            round(cv_results['accuracy'], 4),
            round(cv_results['precision'], 4),
            round(cv_results['recall'], 4),
            round(cv_results['f1'], 4),
            round(cv_results['specificity'], 4),
            round(cv_results['roc_auc'], 4)
        ],
        'Test_Set': [
            round(test_results['accuracy'], 4),
            round(test_results['precision'], 4),
            round(test_results['recall'], 4),
            round(test_results['f1'], 4),
            round(test_results['specificity'], 4),
            round(test_results['roc_auc'], 4)
        ],
        'Unseen_Data': [
            round(unseen_results['accuracy'], 4),
            round(unseen_results['precision'], 4),
            round(unseen_results['recall'], 4),
            round(unseen_results['f1'], 4),
            round(unseen_results['specificity'], 4),
            round(unseen_results['roc_auc'], 4)
        ]
    })

    comparison.to_csv(OUTPUT_PATH, index=False)
    print(f"\n   Results saved to {OUTPUT_PATH}")

    # Print formatted results table
    print("\n" + "=" * 70)
    print("SVM RESULTS SUMMARY")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print("=" * 70)


def main():
    print("\n" + "=" * 60)
    print("SUPPORT VECTOR MACHINE - STROKE PREDICTION")
    print("=" * 60)

    # 1. Train the model with correct pipeline order
    print("\n   Training SVM model...")
    model, scaler, y_test_pred, y_test_proba, y_train_resampled = train_model()

    # 2. Test Set Evaluation
    print("\n   Evaluating on TEST set...")
    test_results = evaluate_metrics(y_test, y_test_pred, y_test_proba)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Generate plots for test set
    plot_confusion_matrix(cm_test, "test")
    plot_roc_curve(y_test, y_test_proba, "test")

    # 3. Unseen Set Evaluation
    print("\n   Evaluating on UNSEEN set...")
    X_unseen_scaled = scaler.transform(X_unseen)
    y_unseen_pred = model.predict(X_unseen_scaled)
    y_unseen_proba = model.predict_proba(X_unseen_scaled)[:, 1]

    unseen_results = evaluate_metrics(y_unseen, y_unseen_pred, y_unseen_proba)
    cm_unseen = confusion_matrix(y_unseen, y_unseen_pred)

    # Generate plots for unseen set
    plot_confusion_matrix(cm_unseen, "unseen")
    plot_roc_curve(y_unseen, y_unseen_proba, "unseen")

    # 4. Cross-Validation (using correct pipeline order)
    print("\n   Performing 10-fold cross-validation...")
    cv_results = cross_validate_svm(X_train, y_train)

    # 5. Save all results
    save_results_to_file(test_results, unseen_results, cv_results)

    # 6. Print classification reports
    print("\n   Detailed Classification Report - TEST SET:")
    print(classification_report(y_test, y_test_pred, target_names=['No Stroke', 'Stroke']))

    print("\n   Detailed Classification Report - UNSEEN SET:")
    print(classification_report(y_unseen, y_unseen_pred, target_names=['No Stroke', 'Stroke']))

    print("\n   SVM Analysis Complete!")
    print(f"   All outputs saved to: {OUTPUT_DIR}")


# ======================================
# RUN THE SCRIPT
# ======================================

if __name__ == "__main__":
    main()