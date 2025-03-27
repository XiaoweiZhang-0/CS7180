import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('tiktok_dataset_with_engagement_score.csv')

# Feature matrix X (including hashtagFreqFeature)
X = df[[
    'video.duration',
    'desc_len',
    'num_hashtags',
    'verified',
    'hour',
    'weekday',
    'is_original_sound',
    'hashtagFreqFeature',
    'stats.diggCount',
    'stats.commentCount',
    'stats.shareCount'
]]

# Target variable y
y = df['engagement_score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 2. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# 3. Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Evaluate models
def evaluate(model_name, y_true, y_pred):
    print(f"\n📊 {model_name} Performance:")
    print(f"  R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"  MSE: {mean_squared_error(y_true, y_pred):.4f}")

evaluate("Random Forest", y_test, rf_pred)
evaluate("Linear Regression", y_test, lr_pred)
evaluate("Gradient Boosting", y_test, gb_pred)

print("\n✅ All models trained and evaluated")
