# # ======================================
# Programmer: Nathaniel de la Rosa
# Date Programmed: April 02, 2026

# Objectives:
# Model 2 – Naive Bayes
# Responsible for implementing a probabilistic classification model.

# Tasks performed:
# Load the processed dataset
# Split data into training, validation, and test sets
# Train a Gaussian Naive Bayes model
# Generate predictions on validation and test sets
# Evaluate model performance using classification report
# Save results to outputs/naive_bayes_results.txt

# Outputs:
# outputs/naive_bayes_results.txt
# # ======================================

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_stroke_data.csv"
OUTPUT_DIR = BASE_DIR / "notebooks" / "outputs" # I'll modify once the folder structure is finalized

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Note: I think train-test-val is supposed to be the same across all the models.
# Remove this code block and replace with calls to folders containing the split data.
# 1. First split: 90% train+val, 10% test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# 2. Second split: 80% of X_train_val goes to train, 20% to validation
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)


# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on validation
y_val_pred = model.predict(X_val)

# Predict on unseen data
y_test_pred = model.predict(X_test)

# Generate reports
val_report = classification_report(y_val, y_val_pred)
test_report = classification_report(y_test, y_test_pred)

# Save results
results_path = OUTPUT_DIR / "naive_bayes_results.txt"
with open(results_path, "w") as f:
    f.write("Naive Bayes Validation Results\n")
    f.write("==============================\n")
    f.write(val_report + "\n\n")

    f.write("Naive Bayes Unseen Data Results\n")
    f.write("===============================\n")
    f.write(test_report)

print(f"Results saved to: {results_path}")