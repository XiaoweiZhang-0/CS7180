import pandas as pd

df = pd.read_csv('tiktok_dataset.csv')

df['desc_len'] = df['desc'].astype(str).apply(len)

df['num_hashtags'] = df['desc'].astype(str).str.count('#')

df['verified'] = df['author.verified'].astype(int)

df['hour'] = pd.to_datetime(df['createTime']).dt.hour
df['weekday'] = pd.to_datetime(df['createTime']).dt.weekday

df['is_original_sound'] = df['music.title'].astype(str).apply(lambda x: int('original sound' in x.lower()))

X = df[[
    'video.duration',
    'desc_len',
    'num_hashtags',
    'verified',
    'hour',
    'weekday',
    'is_original_sound'
]]

# build a target variable y by combining 4 features with different weights
from sklearn.preprocessing import MinMaxScaler

engagement_cols = [
    'stats.playCount',
    'stats.diggCount',
    'stats.commentCount',
    'stats.shareCount'
]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[engagement_cols]), columns=['play', 'digg', 'comment', 'share'])

df['engagement_score'] = (
    0.4 * df_scaled['play'] +
    0.3 * df_scaled['digg'] +
    0.2 * df_scaled['comment'] +
    0.1 * df_scaled['share']
)

y = df['engagement_score']

df.to_csv('tiktok_dataset_with_engagement_score.csv', index=False)
# X.to_csv('X_features.csv', index=False)
# y.to_csv('y_target.csv', index=False)

print("✅")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")