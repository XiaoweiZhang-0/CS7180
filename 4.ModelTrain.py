import pickle

import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# === 1. Load dataset ===
# df = pd.read_csv('tiktok_dataset_with_engagement_score.csv')

X = pd.read_csv("X_features.csv")
y = pd.read_csv("y_target.csv")


le = LabelEncoder()
y["engagement_category"] = le.fit_transform(
    y["engagement_category"]
)  # Ensure y is encoded if categorical

# Remove 'resolution' column as it contains non-numeric values ('720p', '480p')
if "resolution" in X.columns:
    X = X.drop(columns=["resolution"])
    print("Removed 'resolution' column")

# Convert y to a 1D array
y = y.values.ravel()
print(f"Converted y to 1D array with shape: {y.shape}")

# === 2. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# === 3. Evaluation helper ===
def evaluate_model(model_name, y_true, y_pred):
    ## Evaluate the model's performance using precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Model: {model_name}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


# === 4. Initialize and evaluate models ===
results = []

## 1. gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train.ravel())  # ravel to convert y_train to 1D array
y_pred_gb = gb_model.predict(X_test)
results.append(evaluate_model("Gradient Boosting Classifier", y_test, y_pred_gb))

## 2. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train.ravel())  # ravel to convert y_train to 1D array
y_pred_rf = rf_model.predict(X_test)
results.append(evaluate_model("Random Forest Classifier", y_test, y_pred_rf))

## 3. XGBoost Classifier
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    random_state=42, use_label_encoder=False, eval_metric="mlogloss"
)
xgb_model.fit(X_train, y_train.ravel())  # ravel to convert y_train to 1D array

# Save the model to pickle file
with open("model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("✅ Model has been saved to model.pkl")

results.append(evaluate_model("StackingRegressor", y_test, xgb_model.predict(X_test)))
