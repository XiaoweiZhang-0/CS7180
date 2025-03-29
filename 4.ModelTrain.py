import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV

# === 1. Load dataset ===
# df = pd.read_csv('tiktok_dataset_with_engagement_score.csv')

X = pd.read_csv('X_features.csv')
y = pd.read_csv('y_target.csv')

# === 2. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Evaluation helper ===
def evaluate_model(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        'Model': name,
        'MSE': round(mse, 4),
        'MAE': round(mae, 4),
        'R²': round(r2, 4)
    }

# === 4. Initialize and evaluate models ===
results = []

# # (1) Gradient Boosting
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(X_train, y_train)
# results.append(evaluate_model("GradientBoosting", y_test, gb_model.predict(X_test)))

# # (2) Random Forest
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
# results.append(evaluate_model("RandomForest", y_test, rf_model.predict(X_test)))

# # (3) Combined: GradientBoosting + RandomForest not directly combinable, so we just compare them separately

# (4) Stacking: Decision Tree + Random Forest -> RidgeCV
stack_model = StackingRegressor(
    estimators=[
        ('dt', DecisionTreeRegressor(random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42))
    ],
    final_estimator=RidgeCV(),
    passthrough=True
)
stack_model.fit(X_train, y_train)
results.append(evaluate_model("StackingRegressor", y_test, stack_model.predict(X_test)))

# # (5) HistGradientBoosting (Fastest native boosting in sklearn)
# hgb_model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
# hgb_model.fit(X_train, y_train)
# results.append(evaluate_model("HistGradientBoosting", y_test, hgb_model.predict(X_test)))

# === 5. Show comparison result ===
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison Results:")
print(results_df.sort_values(by='R²', ascending=False))