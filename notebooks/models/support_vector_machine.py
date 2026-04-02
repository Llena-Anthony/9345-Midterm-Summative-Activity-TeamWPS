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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============================================================================
# PATHS - Define the path to your data file
# ============================================================================

# Get the current script's directory
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent

# Path to your processed data file
DATA_PATH = SCRIPT_DIR / "data" / "processed" / "cleaned_stroke_data.csv"

#Path to store the result
OUTPUT_PATH = SCRIPT_DIR / "data" / "outputs" / "svm_results.txt"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

ROC_IMG_PATH = SCRIPT_DIR / "data" / "outputs" / "roc_curve.png"

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    """Load the stroke dataset"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f" Data loaded from: {DATA_PATH}")
    print(f" Shape: {df.shape}")
    print(f" Columns: {df.columns.tolist()}")
    return df


# ============================================================================
# TRAIN MODEL
# ============================================================================

def train_model(df):
    """Train the SVM model"""
    print("\n" + "=" * 60)
    print("TRAINING SVM MODEL")
    print("=" * 60)

    # Separate features and target
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Stroke percentage: {y.mean() * 100:.2f}%")

    # Handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\nClass weights: {class_weight_dict}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight=class_weight_dict,
        random_state=42,
        probability=True
    )

    svm_model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = svm_model.predict(X_test_scaled)
    y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n Model Performance:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Write results to file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("SVM STROKE PREDICTION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC: {roc:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(report + "\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    print(f"\n Results saved to: {OUTPUT_PATH}")

    return svm_model, scaler, X.columns, X_test_scaled, y_test, y_pred_proba


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_stroke(model, scaler, feature_names, patient_data):
    """
    Predict stroke for a single patient

    Args:
        model: Trained SVM model
        scaler: Fitted StandardScaler
        feature_names: List of feature names in correct order
        patient_data: Dictionary with patient features (use original column names)

    Returns:
        prediction: 0 or 1 (no stroke or stroke)
        probability: Probability of stroke (0-1)
    """
    # Create feature array in the correct order
    features = np.array([[patient_data[feature] for feature in feature_names]])

    # Convert to DataFrame with column names to avoid warnings
    features_df = pd.DataFrame(features, columns=feature_names)

    # Scale features
    features_scaled = scaler.transform(features_df)

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return prediction, probability


# ============================================================================
# EXAMPLE PREDICTIONS
# ============================================================================

def test_predictions(model, scaler, feature_names, df):
    """Test predictions on sample data"""
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    # Calculate a better threshold based on data distribution
    # For stroke prediction, use a lower threshold since stroke is rare
    risk_thresholds = {
        'high': 0.3,  # 30% or higher
        'moderate': 0.15,  # 15-30%
        'low': 0.05  # 5-15%
    }

    # Test on first 5 rows of the dataset
    for i in range(min(5, len(df))):
        sample = df.iloc[i]

        # Create patient data dictionary
        patient = {col: sample[col] for col in feature_names}

        # Make prediction
        pred, prob = predict_stroke(model, scaler, feature_names, patient)

        print(f"\nPatient {i + 1}:")
        print(f"  Age: {sample['age']}, BMI: {sample['bmi']:.1f}, Glucose: {sample['avg_glucose_level']:.1f}")
        print(f"  Hypertension: {sample['hypertension']}, Heart Disease: {sample['heart_disease']}")
        print(f"  Actual stroke: {sample['stroke']}")
        print(f"  Predicted stroke: {pred}")
        print(f"  Probability of stroke: {prob:.4f}")

        # Updated risk assessment with more appropriate thresholds
        if prob >= risk_thresholds['high']:
            print(f"  HIGH RISK - Probability {prob:.1%}")
        elif prob >= risk_thresholds['moderate']:
            print(f"  MODERATE RISK - Probability {prob:.1%}")
        elif prob >= risk_thresholds['low']:
            print(f"  LOW RISK - Probability {prob:.1%}")
        else:
            print(f"  VERY LOW RISK - Probability {prob:.1%}")

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
    plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_score:.2f})", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # random baseline

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Stroke Prediction")
    plt.legend(loc="lower right")

    # Save image
    plt.savefig(ROC_IMG_PATH)
    plt.close()

    print(f" ROC curve saved to: {ROC_IMG_PATH}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 60)
    print("STROKE PREDICTION SVM MODEL")
    print("=" * 60)

    try:
        # Load data
        df = load_data()

        # Train model
        model, scaler, feature_names, X_test_scaled, y_test, y_pred_proba = train_model(df)

        plot_roc_curve(y_test, y_pred_proba)

        # Test predictions
        test_predictions(model, scaler, feature_names, df)

        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETE!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease make sure your data file is at:")
        print(f"  {DATA_PATH}")
        print("\nYour project structure should look like:")
        print("  your_project/")
        print("  ├── stroke_prediction.py  (this file)")
        print("  └── data/")
        print("      └── processed/")
        print("          └── cleaned_stroke_data.csv")

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()