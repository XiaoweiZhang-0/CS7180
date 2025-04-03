import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import ast


# Load preprocessed dataset
df = pd.read_csv('tiktok_dataset.csv')

# Feature: desc_length
df['desc_length'] = df['desc'].astype(str).apply(len)

# Feature: has_trending_hashtag / num_trending_hashtags
df['hashtags'] = df['hashtags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# flatten all hashtags to lowercase
all_hashtags = [tag.lower() for tags in df['hashtags'] for tag in tags]
hashtag_counts = Counter(all_hashtags)
TOP_PERCENT = 0.05
top_n = max(1, int(len(hashtag_counts) * TOP_PERCENT))
popular_hashtags = set([tag for tag, _ in hashtag_counts.most_common(top_n)])

def count_popular_hashtags(tags):
    tags_lower = [tag.lower() for tag in tags]
    num = sum(1 for tag in tags_lower if tag in popular_hashtags)
    return pd.Series({
        "has_trending_hashtag": int(num > 0),
        "num_trending_hashtags": num
    })

df[['has_trending_hashtag', 'num_trending_hashtags']] = df['hashtags'].apply(count_popular_hashtags)

# Feature: is_trending_music
music_counts = df['music_title'].value_counts()
music_top_n = max(1, int(len(music_counts) * TOP_PERCENT))
popular_music_titles = set(music_counts.head(music_top_n).index)
df['is_trending_music'] = df['music_title'].apply(lambda x: int(x in popular_music_titles))

# Feature: has_mention
def has_mention(text_extra):
    try:
        parsed = json.loads(text_extra.replace("'", '"'))
        return int(any(tag.get("userUniqueId") for tag in parsed if isinstance(tag, dict)))
    except:
        return 0

df['has_mention'] = df['textExtra'].apply(has_mention)

# Feature: hour_posted
df['hour_posted'] = pd.to_datetime(df['createTime']).dt.hour

# Convert verified to int
df['verified'] = df['author.verified'].astype(int)

df['music_id'] = df['music.id'].astype(str)

# TODO:
# 1. Description: change length to transformer evaluated attitude: positive, negative, neutral
# 2. Music: music id
# 3. Aspect ratio: video.width, video.height
# 4. Video resolution

# Feature matrix X with added hashtagFreqFeature
X = df[[
    'desc_length',
    'has_trending_hashtag',
    'num_trending_hashtags',
    'is_trending_music',
    'has_mention',
    'hour_posted',
    'video_duration',

    'verified',
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
