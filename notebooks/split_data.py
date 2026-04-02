# ======================================
# Programmer: Anthony Llena
# Task: Unified Data Splitting
# ======================================

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_stroke_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# ======================================
# 1. Split → 90% (train+test) / 10% (unseen)
# ======================================
X_train_test, X_unseen, y_train_test, y_unseen = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

# ======================================
# 2. Split → 80% train / 20% test
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X_train_test, y_train_test,
    test_size=0.20,
    random_state=42,
    stratify=y_train_test
)

# ======================================
# SAVE FILES
# ======================================
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
X_unseen.to_csv(OUTPUT_DIR / "X_unseen.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)
y_unseen.to_csv(OUTPUT_DIR / "y_unseen.csv", index=False)

print("Data successfully split and saved!")