import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load preprocessed dataset
df = pd.read_csv('tiktok_dataset.csv')

# Print available columns for debugging
print("\nAvailable columns:")
print(df.columns.tolist())

# Feature: length of description
df['desc_len'] = df['desc'].astype(str).apply(len)

# Feature: number of hashtags in description
df['num_hashtags'] = df['desc'].astype(str).str.count('#')

# Author features
df['author_verified'] = df['author.verified'].astype(int)
df['author_signature_len'] = df['author.signature'].astype(str).apply(len)

# Music features
df['music_original'] = df['music.original'].astype(int)
df['music_title_len'] = df['music.title'].astype(str).apply(len)

# Video features
df['video_duration'] = df['video.duration'].astype(float)
df['video_ratio'] = df['video.ratio'].astype(float)
df['video_width'] = df['video.width'].astype(float)
df['video_height'] = df['video.height'].astype(float)

# Extract hour and weekday from createTime
df['hour'] = pd.to_datetime(df['createTime']).dt.hour
df['weekday'] = pd.to_datetime(df['createTime']).dt.weekday

# Feature matrix X with all features
X = df[[
    'video_duration',
    'video_ratio',
    'video_width',
    'video_height',
    'desc_len',
    'num_hashtags',
    'author_verified',
    'author_signature_len',
    'music_original',
    'music_title_len',
    'hour',
    'weekday'
]]

# Target variable (using a simple engagement score based on available metrics)
df['engagement_score'] = (
    0.3 * df['author_verified'] +
    0.2 * df['music_original'] +
    0.2 * df['desc_len'] +
    0.15 * df['num_hashtags'] +
    0.15 * df['author_signature_len']
)

# Target variable
y = df['engagement_score']

# Save updated dataset with engagement score
df.to_csv('tiktok_dataset_with_engagement_score.csv', index=False)

print("✅")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("\nFeature names:")
print(X.columns.tolist())
