# naive_bayes_pipeline.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ==========================
# Paths
# ==========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_stroke_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# ==========================
# Load dataset
# ==========================
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# ==========================
# Train/Test split (stratified)
# ==========================
# Note: I think train-test-val is supposed to be the same across all the models. Remove this and replace with
# calls to folders containing the split data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# Train Gaussian Naive Bayes
# ==========================
model = GaussianNB()
model.fit(X_train, y_train)

# ==========================
# Predictions & Metrics
# ==========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# ==========================
# Save metrics to text file
# ==========================
results_path = OUTPUT_DIR / "naive_bayes_results.txt"
with open(results_path, "w") as f:
    f.write("Naive Bayes Stroke Prediction Results\n")
    f.write("===============================\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Weighted F1 Score: {f1:.4f}\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
print(f"Results saved to: {results_path}")

# ==========================
# Save confusion matrix plot
# ==========================
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)
plt.title("Naive Bayes Confusion Matrix")
cm_path = OUTPUT_DIR / "naive_bayes_confusion_matrix.png"
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved to: {cm_path}")