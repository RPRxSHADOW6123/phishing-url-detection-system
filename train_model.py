import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from feature_extraction import extract_features

file_path = "dataset/phishing_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Total rows:", len(df))

df = df[["URL", "label"]]

df = df.dropna()

df["label"] = df["label"].map({
    "benign": 0,
    "phishing": 1
})

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

df = df.sample(
    n=min(60000, len(df)),
    random_state=42
)

print("Samples used for training:", len(df))

print("\nExtracting URL features...")

feature_data = df["URL"].apply(extract_features)

X = pd.DataFrame(feature_data.tolist())

y = df["label"]


print("\nFeatures extracted:")
print(X.head())

print("\nFeature columns:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining Gradient Boosting Classifier...")

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Phishing"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model_path = "model/phishing_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved at:", model_path)