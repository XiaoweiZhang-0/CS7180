import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed dataset
df = pd.read_csv('tiktok_dataset.csv')

# Feature: length of description
df['desc_len'] = df['desc'].astype(str).apply(len)

# Feature: number of hashtags in description
df['num_hashtags'] = df['desc'].astype(str).str.count('#')

# Convert verified to int
df['verified'] = df['author.verified'].astype(int)

# Extract hour and weekday from createTime
df['hour'] = pd.to_datetime(df['createTime']).dt.hour
df['weekday'] = pd.to_datetime(df['createTime']).dt.weekday

# Flag if music title includes 'original sound'
df['is_original_sound'] = df['music.title'].astype(str).apply(lambda x: int('original sound' in x.lower()))

# Feature matrix X with added hashtagFreqFeature
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

# Engagement-related columns
engagement_cols = [
    'stats.playCount',
    'stats.diggCount',
    'stats.commentCount',
    'stats.shareCount'
]

# Normalize engagement columns
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[engagement_cols]), columns=['play', 'digg', 'comment', 'share'])

# Compute weighted engagement score
df['engagement_score'] = (
    0.4 * df_scaled['play'] +
    0.3 * df_scaled['digg'] +
    0.2 * df_scaled['comment'] +
    0.1 * df_scaled['share']
)

# Target variable
y = df['engagement_score']

# Save updated dataset with engagement score
df.to_csv('tiktok_dataset_with_engagement_score.csv', index=False)

print("✅")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
