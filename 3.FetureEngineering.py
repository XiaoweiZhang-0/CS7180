import ast
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from numpy import hstack as np_hstack
from scipy.sparse import save_npz

from text_embedding import bert, tf_idf

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


# Use text embedding technique in text_embedding.py to transform desc into a feature vector
# tfidf_matrix_desc = tf_idf("subset_10000.json", "desc") # Implement on subset to explore data
# tf_idf_matrix_author_signature = tf_idf(
#     "subset_10000.json", "author.signature"
# )  # Implement on subset to explore data
# tf_idf_matrix_author_nickname = tf_idf(
#     "subset_10000.json", "author.nickname"
# )  # Implement on subset to explore data
# tf_idf_matrix_challenges = tf_idf(
#     "subset_10000.json", "challenges"
# )  # Implement on subset to explore data
# # merge the four matrices
# tfidf_matrix = hstack(
#     [
#         tfidf_matrix_desc,
#         tf_idf_matrix_author_signature,
#         tf_idf_matrix_author_nickname,
#         tf_idf_matrix_challenges,
#     ]
# )

bert_matrix_desc = bert("subset_10000.json", "desc") # Implement on subset to explore data
bert_matrix_author_signature = bert(
    "subset_10000.json", "author.signature"
)  # Implement on subset to explore data
bert_matrix_author_nickname = bert(
    "subset_10000.json", "author.nickname"
)  # Implement on subset to explore data
bert_matrix_challenges = bert(  
    "subset_10000.json", "challenges"
)  # Implement on subset to explore data

# merge the four matrices
bert_matrix = np_hstack(
    [
        bert_matrix_desc,
        bert_matrix_author_signature,
        bert_matrix_author_nickname,
        bert_matrix_challenges,
    ]
)

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
for column in X.columns:
    if X[column].dtype == "object":
        # For categorical columns, fill with the mode
        X[column].fillna(X[column].mode()[0], inplace=True)
    elif X[column].dtype in ["float64", "int64"]:
        # For numeric columns, fill with the median
        X[column].fillna(X[column].median(), inplace=True)
X_sparse = csr_matrix(X)
# X = hstack([tfidf_matrix, X_sparse])
X = np_hstack([X, bert_matrix])

# X = pd.concat([X.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)




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
# df.to_csv("tiktok_dataset_with_engagement_score.csv", index=False)
# X.to_csv("X_features.csv", index=False)
# save_npz('X.npz', X)
np.save('X.npy', X)

y.to_csv("y_target.csv", index=False)

print("✅")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
