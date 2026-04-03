# ======================================
# Programmer: Nathaniel de la Rosa
# Date Programmed: April 02, 2026
# Model 2 – Naive Bayes with SMOTE & 10-fold CV
# Objectives:
# - Implement a probabilistic classification model (Gaussian Naive Bayes)
# - Apply SMOTE to balance training data
# - Perform 10-fold cross-validation
# - Evaluate using confusion matrix, accuracy, precision, recall, F1-score, specificity, ROC-AUC
# - Save confusion matrices and metrics results
# Outputs:
# - outputs/confusion_matrix_test.png
# - outputs/confusion_matrix_unseen.png
# - outputs/naive_bayes_results.csv
# ======================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, make_scorer)
from imblearn.over_sampling import SMOTE

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "naive_bayes"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
X_unseen = pd.read_csv(DATA_DIR / "X_unseen.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
y_unseen = pd.read_csv(DATA_DIR / "y_unseen.csv").squeeze()

# Apply SMOTE
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Define model
model = GaussianNB()

# Metrics
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'specificity': make_scorer(specificity_score),
    'roc_auc': make_scorer(roc_auc_score)
}

# 10-fold CV
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(model, X_train_res, y_train_res, cv=kf, scoring=scoring)

# Train final model
model.fit(X_train_res, y_train_res)

# Evaluation
def evaluate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "specificity": tn / (tn + fp),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }


def plot_confusion_matrix(y_true, y_pred, name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    plt.title(f'Confusion Matrix - {name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Save to OUTPUT_DIR
    file_path = OUTPUT_DIR / f'confusion_matrix_{name}.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    return file_path


# Predictions
y_test_pred = model.predict(X_test)
y_unseen_pred = model.predict(X_unseen)

# Compute metrics
test_metrics = evaluate_metrics(y_test, y_test_pred)
unseen_metrics = evaluate_metrics(y_unseen, y_unseen_pred)

# Save confusion matrices
test_cm_path = plot_confusion_matrix(y_test, y_test_pred, "test")
unseen_cm_path = plot_confusion_matrix(y_unseen, y_unseen_pred, "unseen")

# Save CSV results
metrics_list = list(scoring.keys())
results_df = pd.DataFrame({
    "Metric": metrics_list,
    "CV (Training)": [np.mean(cv_results[f"test_{m}"]) for m in metrics_list],
    "Test Set": [test_metrics[m] for m in metrics_list],
    "Unseen Set": [unseen_metrics[m] for m in metrics_list]
})
csv_path = OUTPUT_DIR / "naive_bayes_results.csv"
results_df.to_csv(csv_path, index=False)

# Print file locations
print("Successfully completed model evaluation.")
print(f"Confusion Matrix (Test) saved at {test_cm_path}")
print(f"Confusion Matrix (Unseen) saved at {unseen_cm_path}")
print(f"CSV Results saved at {csv_path}")