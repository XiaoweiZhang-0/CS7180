import ast
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed dataset
df = pd.read_csv("tiktok_dataset.csv")

# Feature: desc_length
df["desc_length"] = df["desc"].astype(str).apply(len)


# Feature: has_trending_hashtag / num_trending_hashtags(top 5%)
def extract_hashtags_from_textExtra(text_extra):
    try:
        if pd.isna(text_extra):
            return []
        text_extra_parsed = ast.literal_eval(text_extra)
        return [
            tag.get("hashtagName", "").lower()
            for tag in text_extra_parsed
            if isinstance(tag, dict) and tag.get("type") == 1 and tag.get("hashtagName")
        ]
    except Exception as e:
        return []


df["hashtags"] = df["textExtra"].apply(extract_hashtags_from_textExtra)

# flatten all hashtags to lowercase
all_hashtags = [tag.lower() for tags in df["hashtags"] for tag in tags]
hashtag_counts = Counter(all_hashtags)
TOP_PERCENT = 0.05
top_n = max(1, int(len(hashtag_counts) * TOP_PERCENT))
popular_hashtags = set([tag for tag, _ in hashtag_counts.most_common(top_n)])

# # Save popular hashtags
# with open("popular_hashtags.json", "w") as f:
#     json.dump(list(popular_hashtags), f, ensure_ascii=False, indent=2)

def count_popular_hashtags(tags):
    tags_lower = [tag.lower() for tag in tags]
    num = sum(1 for tag in tags_lower if tag in popular_hashtags)
    return pd.Series(
        {"has_trending_hashtag": int(num > 0), "num_trending_hashtags": num}
    )


df[["has_trending_hashtag", "num_trending_hashtags"]] = df["hashtags"].apply(
    count_popular_hashtags
)


# Feature: is_trending_music(top 5%)
music_counts = df["music.title"].value_counts()
music_top_n = max(1, int(len(music_counts) * TOP_PERCENT))
popular_music_titles = set(music_counts.head(music_top_n).index)
df["is_trending_music"] = df["music.title"].apply(
    lambda x: int(x in popular_music_titles)
)

# # Save popular music
# with open("popular_music_titles.json", "w") as f:
#     json.dump(list(popular_music_titles), f, ensure_ascii=False, indent=2)

# Feature: has_mention
def has_mention(text_extra):
    try:
        parsed = json.loads(text_extra.replace("'", '"'))
        return int(
            any(tag.get("userUniqueId") for tag in parsed if isinstance(tag, dict))
        )
    except:
        return 0


df["has_mention"] = df["textExtra"].apply(has_mention)

# Feature: hour_posted
df["hour_posted"] = pd.to_datetime(df["createTime"]).dt.hour


# Feature matrix X by return the target variables
# and keeping only relevant features
X = df[
    [
        "desc_length",
        "has_trending_hashtag",
        "num_trending_hashtags",
        "is_trending_music",
        "has_mention",
        "hour_posted",
        "video.duration",
        "author.verified",
        "author_signature_len",
        "author_name_len",
        "avg_challenge_desc_length",
        "num_challenges",
        "challenge_completeness",
    ]
]


# Replace NaN values in the feature matrix X
# if x has catergorical or numeric columns with NaN, fill with median for numeric, and mode for categorical
# Fill NaN for categorical columns with mode
X = X.fillna(X.median())


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
