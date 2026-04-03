# ======================================
# Programmer: Nathaniel de la Rosa
# Date Programmed: April 02, 2026
# Model: Gaussian Naive Bayes for Stroke Prediction
# Objectives:
# - Implement Gaussian Naive Bayes for binary classification
# - Perform 10-fold cross-validation
# - Evaluate using confusion matrix, accuracy, precision, recall,
#   F1-score, specificity, and ROC-AUC
# Outputs:
# - confusion_matrix_test.png (test set with 80-20 split)
# - confusion_matrix_unseen.png (unseen 10% holdout data)
# - naive_bayes_results.csv (metrics comparison)
# ======================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "naive_bayes"
OUTPUT_DIR.mkdir(exist_ok=True)

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all metrics for given predictions"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    roc_auc = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'specificity': specificity, 'roc_auc': roc_auc, 'cm': cm
    }

def save_confusion_matrix(cm, title, filename, cmap):
    """Save confusion matrix as PNG with readable labels and proper sizing"""
    plt.figure(figsize=(6, 5))
    cm_df = pd.DataFrame(cm, index=['No Stroke', 'Stroke'], columns=['No Stroke', 'Stroke'])
    sns.heatmap(cm_df, annot=True, fmt='d', cmap=cmap, cbar=True,
                square=True, annot_kws={'size': 14})
    plt.title(title, fontsize=12, pad=15)
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

# Load the pre-split data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
X_unseen = pd.read_csv(DATA_DIR / "X_unseen.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
y_unseen = pd.read_csv(DATA_DIR / "y_unseen.csv").squeeze()

# Initialize Gaussian Naive Bayes model
gnb = GaussianNB()

# 10-fold cross-validation on training data
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(gnb, X_train, y_train, cv=cv, method='predict')
y_cv_pred_proba = cross_val_predict(gnb, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
cv_metrics = calculate_metrics(y_train, y_cv_pred, y_cv_pred_proba)

# Train model and evaluate on test set
gnb.fit(X_train, y_train)
y_test_pred = gnb.predict(X_test)
y_test_pred_proba = gnb.predict_proba(X_test)[:, 1]
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)

# Validate on unseen data
y_unseen_pred = gnb.predict(X_unseen)
y_unseen_pred_proba = gnb.predict_proba(X_unseen)[:, 1]
unseen_metrics = calculate_metrics(y_unseen, y_unseen_pred, y_unseen_pred_proba)

# Save confusion matrices
save_confusion_matrix(test_metrics['cm'], 'Test Set Confusion Matrix (80-20 Split)',
                     'confusion_matrix_test.png', 'Greens')
save_confusion_matrix(unseen_metrics['cm'], 'Unseen Data Confusion Matrix (10% Holdout)',
                     'confusion_matrix_unseen.png', 'Oranges')

# Save combined metrics comparison CSV (rounded to 4 decimals)
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC'],
    'Cross-Validation': [
        round(cv_metrics['accuracy'], 4), round(cv_metrics['precision'], 4),
        round(cv_metrics['recall'], 4), round(cv_metrics['f1'], 4),
        round(cv_metrics['specificity'], 4), round(cv_metrics['roc_auc'], 4)
    ],
    'Test_Set': [
        round(test_metrics['accuracy'], 4), round(test_metrics['precision'], 4),
        round(test_metrics['recall'], 4), round(test_metrics['f1'], 4),
        round(test_metrics['specificity'], 4), round(test_metrics['roc_auc'], 4)
    ],
    'Unseen_Data': [
        round(unseen_metrics['accuracy'], 4), round(unseen_metrics['precision'], 4),
        round(unseen_metrics['recall'], 4), round(unseen_metrics['f1'], 4),
        round(unseen_metrics['specificity'], 4), round(unseen_metrics['roc_auc'], 4)
    ]
})
comparison.to_csv(OUTPUT_DIR / 'naive_bayes_results.csv', index=False)

print(f"done. Outputs saved to: {OUTPUT_DIR}")
print("  - confusion_matrix_test.png")
print("  - confusion_matrix_unseen.png")
print("  - naive_bayes_results.csv")