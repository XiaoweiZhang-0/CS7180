import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed dataset
df = pd.read_csv("tiktok_dataset.csv")

df["aspect_ratio"] = df["video.height"] / df["video.width"]

df["resolution"] = df["video.ratio"]

df["verified"] = df["author.verified"].astype(int)

df["hour"] = pd.to_datetime(df["createTime"]).dt.hour

df["weekday"] = pd.to_datetime(df["createTime"]).dt.weekday

df["music_id"] = df["music.id"].astype(str)


# Feature matrix X by dropping the target variable and unnecessary columns
# and keeping only relevant features
X = df.drop(
    columns=[
        "id",
        "desc",
        "createTime",
        "author.verified",
        "music.id",
        "video.width",
        "video.height",
        "video.ratio",
        "stats.playCount",
        "stats.diggCount",
        "stats.commentCount",
        "stats.shareCount",
        "hashtags",
    ]
)

# Replace NaN values in the feature matrix X
for col in X.select_dtypes(include=["float64", "int64"]).columns:
    if X[col].isnull().any():
        mean_value = X[col].mean()
        X[col].fillna(mean_value, inplace=True)
for col in X.select_dtypes(include=["object"]).columns:
    if X[col].isnull().any():
        mode_value = X[col].mode()[0] if not X[col].mode().empty else ""
        X[col].fillna(mode_value, inplace=True)

# Engagement-related columns
engagement_cols = [
    "stats.playCount",
    "stats.diggCount",
    "stats.commentCount",
    "stats.shareCount",
]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[engagement_cols]),
    columns=["play", "digg", "comment", "share"],
)

df["engagement_score"] = (
    0.4 * df_scaled["play"]
    + 0.3 * df_scaled["digg"]
    + 0.2 * df_scaled["comment"]
    + 0.1 * df_scaled["share"]
)

# Transform the engagement score into four categories: highly popular, popular, average, and low engagement
print("Engagement Score Distribution:")
print(df["engagement_score"].describe())
quantiles = np.percentile(df["engagement_score"].dropna(), [0, 25, 50, 75, 100])
print("Quantile-based bins for engagement score:")
print(quantiles)
if not np.all(np.diff(quantiles) > 0):
    raise ValueError("Quantile bins are not sorted properly.")
bins = quantiles
if len(bins) != 5:
    raise ValueError(
        "There should be exactly 4 bins for categorization (low, average, popular, highly popular)."
    )


labels = ["low", "average", "popular", "highly popular"]
df["engagement_category"] = pd.cut(
    df["engagement_score"], bins=bins, labels=labels, include_lowest=True
)

# Target variable
y = df["engagement_category"].astype(str)  # Convert to string for classification

# Save updated dataset with engagement score
df.to_csv("tiktok_dataset_with_engagement_score.csv", index=False)
X.to_csv("X_features.csv", index=False)
y.to_csv("y_target.csv", index=False)

print("✅")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
