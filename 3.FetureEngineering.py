import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed dataset
df = pd.read_csv('tiktok_dataset.csv')

# df['desc_len'] = df['desc'].astype(str).apply(len)

df['aspect_ratio'] = df['video.height']/['video.width']

df['resolution'] = df['video.ratio']
# df['num_hashtags'] = df['desc'].astype(str).str.count('#')

# Convert verified to int
df['verified'] = df['author.verified'].astype(int)

# Extract hour and weekday from createTime
df['hour'] = pd.to_datetime(df['createTime']).dt.hour
df['weekday'] = pd.to_datetime(df['createTime']).dt.weekday

df['music_id'] = df['music.id'].astype(str)

# TODO:
# 1. Description: change length to transformer evaluated attitude: positive, negative, neutral
# 2. Music: music id
# 3. Aspect ratio: video.width, video.height
# 4. Video resolution

# Feature matrix X with added hashtagFreqFeature
X = df[[
    'video.duration',
    'desc_len',
    'verified',
    'hour',
    'weekday',
    'aspect_ratio',
    'resolution'
    # 'desc_len',
    # 'num_hashtags',
    'verified',
    'hour',
    'weekday',
    'music_id',
    'hashtag_freq_feature',
    # 'is_original_sound'
]]

# Engagement-related columns
engagement_cols = [
    'stats.playCount',
    'stats.diggCount',
    'stats.commentCount',
    'stats.shareCount'
]

########################
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[engagement_cols]), columns=['play', 'digg', 'comment', 'share'])

##########################
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
X.to_csv('X_features.csv', index=False)
y.to_csv('y_target.csv', index=False)

print("✅")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
