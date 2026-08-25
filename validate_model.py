import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from feature_extraction import extract_features


# -----------------------------------
# 1. Load Dataset 2
# -----------------------------------

file_path = "dataset/phishing_dataset1.csv"

df = pd.read_csv(file_path)

print("Dataset 2 loaded successfully!")
print("Total rows:", len(df))


# -----------------------------------
# 2. Keep Only Benign and Phishing
# -----------------------------------

df = df[df["class_label"].isin(["benign", "phishing"])].copy()

print("\nRows used for validation:", len(df))

print("\nClass distribution:")
print(df["class_label"].value_counts())


# -----------------------------------
# 3. Convert Labels
# -----------------------------------

df["target"] = df["class_label"].map({
    "benign": 0,
    "phishing": 1
})


# -----------------------------------
# 4. Extract Same Features
# -----------------------------------

print("\nExtracting URL features...")

feature_data = df["url"].apply(extract_features)

X = pd.DataFrame(feature_data.tolist())

y = df["target"]


print("\nFeatures used:")
print(X.columns.tolist())


# -----------------------------------
# 5. Load Existing Model
# -----------------------------------

model_path = "model/phishing_model.pkl"

model = joblib.load(model_path)

print("\nExisting Gradient Boosting model loaded.")


# -----------------------------------
# 6. Make Predictions
# -----------------------------------

print("\nRunning independent validation...")

y_pred = model.predict(X)


# -----------------------------------
# 7. Evaluate
# -----------------------------------

accuracy = accuracy_score(y, y_pred)

precision = precision_score(y, y_pred)

recall = recall_score(y, y_pred)

f1 = f1_score(y, y_pred)


print("\n===================================")
print("INDEPENDENT VALIDATION RESULTS")
print("===================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# -----------------------------------
# 8. Classification Report
# -----------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y,
        y_pred,
        target_names=["Benign", "Phishing"]
    )
)


# -----------------------------------
# 9. Confusion Matrix
# -----------------------------------

print("\nConfusion Matrix:")

print(confusion_matrix(y, y_pred))
