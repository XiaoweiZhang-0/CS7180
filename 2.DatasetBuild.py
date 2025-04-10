import json
import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Load and flatten JSON
with open('subset_10000.json', 'r') as f:
    data = json.load(f)

# filter out all secret videos
filtered = [entry for entry in data if not entry.get("secret", False) and not entry.get("forFriend", False)]
print(f"✅ Public videos retained: {len(filtered)}")

df = pd.json_normalize(data, sep='.')

# Select features
df_selected = df[[
    # Video
    'id',
    # Author features
    'author.verified',
    'author.uniqueId',
    'author.nickname',
    'author.signature',
    'author.relation',
    # Challenges features
    'challenges',
    # Stats features
    'desc',
    'textExtra',
    'createTime',
    'video.duration',
    
    # Music
    'music.title',
      
    # Popularity
    'stats.playCount',
    'stats.diggCount',
    'stats.commentCount',
    'stats.shareCount'
]].copy()

# Fill missing values with appropriate defaults
df_selected['author.verified'] = df_selected['author.verified'].fillna(False).astype(int)
df_selected['author.relation'] = df_selected['author.relation'].fillna(0)
df_selected['stats.playCount'] = df_selected['stats.playCount'].fillna(0).astype(int)
df_selected['stats.diggCount'] = df_selected['stats.diggCount'].fillna(0).astype(int)
df_selected['stats.commentCount'] = df_selected['stats.commentCount'].fillna(0).astype(int)
df_selected['stats.shareCount'] = df_selected['stats.shareCount'].fillna(0).astype(int)

# 1. Primary Author Features
df_selected['author_signature_len'] = df_selected['author.signature'].astype(str).apply(len)
df_selected['author_name_len'] = df_selected['author.nickname'].astype(str).apply(len)

# 2. Enhanced Challenge Features
def extract_challenge_info(challenges):
    if not isinstance(challenges, list):
        return 0, 0, 0, 0, 0
    
    num_challenges = len(challenges)
    challenges_with_desc = sum(1 for c in challenges if c.get('desc'))
    challenges_with_thumb = sum(1 for c in challenges if c.get('profileThumb'))
    
    # Calculate average description length
    desc_lengths = [len(str(c.get('desc', ''))) for c in challenges]
    avg_desc_length = sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0
    
    # Calculate challenge completeness (percentage of challenges with both desc and thumb)
    complete_challenges = sum(1 for c in challenges if c.get('desc') and c.get('profileThumb'))
    challenge_completeness = complete_challenges / num_challenges if num_challenges > 0 else 0
    
    return num_challenges, challenges_with_desc, challenges_with_thumb, avg_desc_length, challenge_completeness

# Extract challenge features
challenge_features = df_selected['challenges'].apply(extract_challenge_info)
df_selected['num_challenges'] = challenge_features.apply(lambda x: x[0])
df_selected['challenges_with_desc'] = challenge_features.apply(lambda x: x[1])
df_selected['challenges_with_thumb'] = challenge_features.apply(lambda x: x[2])
df_selected['avg_challenge_desc_length'] = challenge_features.apply(lambda x: x[3])
df_selected['challenge_completeness'] = challenge_features.apply(lambda x: x[4])

# 3. Engagement Score
# Create a DataFrame with just the stats columns for scaling
stats_df = df_selected[['stats.playCount', 'stats.diggCount', 'stats.commentCount', 'stats.shareCount']]
stats_df.columns = ['play', 'digg', 'comment', 'share']

# Scale the stats
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(stats_df),
    columns=stats_df.columns
)

# Calculate engagement score with specified weights
df_selected['engagement_score'] = (
    0.4 * df_scaled['play'] +
    0.3 * df_scaled['digg'] +
    0.2 * df_scaled['comment'] +
    0.1 * df_scaled['share']
)

# Fill any remaining missing values with appropriate defaults
numeric_columns = df_selected.select_dtypes(include=[np.number]).columns
df_selected[numeric_columns] = df_selected[numeric_columns].fillna(df_selected[numeric_columns].median())

# 4. Analyze feature relationships with engagement score
features_for_analysis = [
    'author.verified',
    'author_signature_len',
    'author_name_len',
    'num_challenges',
    'challenges_with_desc',
    'challenges_with_thumb',
    'avg_challenge_desc_length',
    'challenge_completeness'
]

# Calculate correlations
correlations = df_selected[features_for_analysis + ['engagement_score']].corr()['engagement_score'].sort_values(ascending=False)
print("\nFeature Correlations with Engagement Score:")
print(correlations)

# Calculate Mutual Information
X = df_selected[features_for_analysis]
y = df_selected['engagement_score']
mi_scores = mutual_info_regression(X, y)
mi_series = pd.Series(mi_scores, index=features_for_analysis).sort_values(ascending=False)
print("\nMutual Information with Engagement Score:")
print(mi_series)


print("\nSelected features:")
print(df_selected.columns.tolist())

# Save to CSV
df_selected.to_csv('tiktok_dataset.csv', index=False)
print("✅ Raw dataset saved as tiktok_dataset.csv")
print(df_selected.head(3))