# # ======================================
# Programmer: Nathaniel de la Rosa
# Date Programmed: April 02, 2026

# Objectives:
# Model 2 – Naive Bayes
# Responsible for implementing a probabilistic classification model.

# Tasks performed:
# Load the official dataset splits
# Apply SMOTE to handle class imbalance on training data
# Train a Gaussian Naive Bayes model
# Generate predictions on test and unseen sets
# Evaluate model performance using classification report
# Save results to outputs/naive_bayes_results.txt

# Outputs:
# outputs/naive_bayes_results.txt
# # ======================================

from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

# Load splits
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
X_unseen = pd.read_csv(DATA_DIR / "X_unseen.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()
y_unseen = pd.read_csv(DATA_DIR / "y_unseen.csv").values.ravel()

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train_res, y_train_res)

# Predict on test and unseen sets
y_test_pred = model.predict(X_test)
y_unseen_pred = model.predict(X_unseen)

# Generate classification reports
test_report = classification_report(y_test, y_test_pred)
unseen_report = classification_report(y_unseen, y_unseen_pred)

# Save results
results_path = OUTPUT_DIR / "naive_bayes_results.txt"
with open(results_path, "w") as f:
    f.write("Naive Bayes Test Set Results\n")
    f.write("============================\n")
    f.write(test_report + "\n\n")

    f.write("Naive Bayes Unseen Set Results\n")
    f.write("==============================\n")
    f.write(unseen_report)

print(f"Results saved to: {results_path}")