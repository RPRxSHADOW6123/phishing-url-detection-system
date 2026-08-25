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


# -----------------------------------
# 1. Load Dataset 2
# -----------------------------------

file_path = "dataset/phishing_dataset1.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Total rows:", len(df))


# -----------------------------------
# 2. Keep Benign and Phishing
# -----------------------------------

df = df[df["class_label"].isin(["benign", "phishing"])].copy()

print("\nRows used:", len(df))

print("\nClass distribution:")
print(df["class_label"].value_counts())


# -----------------------------------
# 3. Create Target
# -----------------------------------

df["target"] = df["class_label"].map({
    "benign": 0,
    "phishing": 1
})


# -----------------------------------
# 4. Select Engineered Features
# -----------------------------------

features = [
    "domain_len",
    "path_len",
    "query_len",
    "path_depth",
    "subdomain_count",
    "query_param_count",
    "has_port",
    "is_ip_host",
    "has_punycode",
    "is_https",
    "is_shortener",
    "digit_count_v2",
    "letter_count_v2",
    "special_count_v2",
    "digit_ratio",
    "letter_ratio",
    "special_ratio",
    "token_count",
    "max_token_len",
    "avg_token_len",
    "url_entropy",
    "host_entropy",
    "url_len_v2",
    "abnormal_url",
    "having_ip_address",
    "phish_urgency_words",
    "phish_security_words",
    "phish_brand_mentions",
    "phish_brand_hijack",
    "phish_multiple_subdomains",
    "phish_long_path",
    "phish_many_params",
    "phish_suspicious_tld",
    "phish_adv_hyphen_count",
    "phish_adv_number_count",
    "phish_adv_long_domain",
    "phish_adv_many_subdomains",
    "phish_adv_encoded_chars",
    "phish_adv_path_keywords",
    "path_has_hacked_terms",
    "suspicious_extension",
    "path_underscore_count",
    "is_gov_edu"
]


# -----------------------------------
# 5. Prepare Data
# -----------------------------------

X = df[features].copy()
y = df["target"]


# Replace missing/infinite values
X = X.replace([float("inf"), float("-inf")], 0)
X = X.fillna(0)


print("\nNumber of features:", len(features))


# -----------------------------------
# 6. Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------------
# 7. Gradient Boosting Classifier
# -----------------------------------

print("\nTraining Gradient Boosting Classifier...")

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("Training completed!")


# -----------------------------------
# 8. Predictions
# -----------------------------------

y_pred = model.predict(X_test)


# -----------------------------------
# 9. Evaluation
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n===================================")
print("GRADIENT BOOSTING V2 RESULTS")
print("===================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Phishing"]
    )
)


print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


# -----------------------------------
# 10. Save V2 Model
# -----------------------------------

model_path = "model/phishing_model_v2.pkl"

joblib.dump(model, model_path)

print("\nV2 model saved successfully!")
print("Saved at:", model_path)