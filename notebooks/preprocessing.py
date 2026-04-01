# # ======================================
# Programmer: Anthony Llena
# Date Programmed: April 04, 2026
#
# Objectives:
#   This script is responsible for preparing the raw healthcare stroke dataset for machine learning.
#   It performs data cleaning and transformation steps to ensure the dataset is suitable for model training.
#
# Key Tasks Performed:
#   1. Load the raw dataset from the data/raw directory
#   2. Remove unnecessary columns (id)
#   3. Handle missing values
#   4. Encode categorical variables into numerical format
# # ======================================

import pandas as pd
from pathlib import Path


# # ======================================
# Path Configuration
# # ======================================


# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Input (raw data)
RAW_PATH = BASE_DIR / "data" / "raw" / "healthcare-dataset-stroke-data.csv"

# Output (processed data)
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "cleaned_stroke_data.csv"


# # ======================================
# Preprocessing Method
# # ======================================

def preprocess_data():
    """
        This function performs all preprocessing steps:
        1. Load raw dataset
        2. Clean data (drop unnecessary columns, handle missing values)
        3. Encode categorical variables
        4. Save cleaned dataset to processed folder
        """
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    # ----------------------------------------
    # Load Data
    #-----------------------------------------
    print("Loading dataset...")
    df = pd.read_csv(RAW_PATH)

    # Display initial info (for checking)
    print("\nInitial Dataset Info:")
    df.info()
    print(f"\nOriginal shape: {df.shape}")

    # ----------------------------------------
    # Drop Unnecessary Column
    # ----------------------------------------
    print("\nDropping 'id' column...")
    df.drop(columns=["id"], inplace=True)

    # ----------------------------------------
    # Handle Missing Values
    # ----------------------------------------
    print("\nHandling missing values in 'bmi'...")
    df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

    # ----------------------------------------
    # Encode Categorical Variables
    # ----------------------------------------
    categorical_columns = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status"
    ]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    bool_columns = df.select_dtypes(include=["bool"]).columns
    df[bool_columns] = df[bool_columns].astype(int)

    # ----------------------------------------
    # Save Preprocessed Data
    # ----------------------------------------
    print("\nSaving preprocessed dataset...")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("\nPreprocessing Complete!")
    print(f"Processed file saved at: {PROCESSED_PATH}")

    print(f"\nProcessed shape: {df.shape}")
    print("\nRemaining missing values:")
    print(df.isnull().sum())

 # ----------------------------------------
 # Run Script Directly
# ----------------------------------------
if __name__ == "__main__":
    preprocess_data()