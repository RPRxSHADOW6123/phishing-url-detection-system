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


# -----------------------------
# 1. Load Dataset
# -----------------------------

file_path = "dataset/phishing_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Total rows:", len(df))


# -----------------------------
# 2. Select Required Columns
# -----------------------------

df = df[["URL", "label"]]

# Remove missing values
df = df.dropna()

# Convert labels to numerical values
df["label"] = df["label"].map({
    "benign": 0,
    "phishing": 1
})

# Remove rows with unknown labels
df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)


# -----------------------------
# 3. Use a Manageable Dataset
# -----------------------------

# Use 60,000 samples for faster training
df = df.sample(
    n=min(60000, len(df)),
    random_state=42
)

print("Samples used for training:", len(df))


# -----------------------------
# 4. Extract URL Features
# -----------------------------

print("\nExtracting URL features...")

feature_data = df["URL"].apply(extract_features)

X = pd.DataFrame(feature_data.tolist())

y = df["label"]


print("\nFeatures extracted:")
print(X.head())

print("\nFeature columns:")
print(X.columns.tolist())


# -----------------------------
# 5. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 6. Create Gradient Boosting Model
# -----------------------------

print("\nTraining Gradient Boosting Classifier...")

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)


# -----------------------------
# 7. Train Model
# -----------------------------

model.fit(X_train, y_train)

print("Model training completed!")


# -----------------------------
# 8. Make Predictions
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# 9. Evaluate Model
# -----------------------------

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


# -----------------------------
# 10. Save Model
# -----------------------------

model_path = "model/phishing_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved at:", model_path)