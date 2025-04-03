import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# load data
with open("subset_10000.json", "r") as f:
    raw_data = json.load(f)

# cal popularity of music/music author
music_titles = [item.get("music", {}).get("title", "") for item in raw_data]
music_authors = [item.get("music", {}).get("authorName", "") for item in raw_data]

title_counts = pd.Series(music_titles).value_counts()
author_counts = pd.Series(music_authors).value_counts()

TOP_PERCENT = 0.05
top_title_n = int(len(title_counts) * TOP_PERCENT)
top_author_n = int(len(author_counts) * TOP_PERCENT)

popular_music_set = set(title_counts.head(top_title_n).index)
popular_author_set = set(author_counts.head(top_author_n).index)

# cal popularity of hashtag
all_hashtags = []
for item in raw_data:
    tags = item.get("textExtra", [])
    for tag in tags:
        if tag.get("type") == 1:
            name = tag.get("hashtagName", "")
            if name:
                all_hashtags.append(name.lower())

hashtag_counts = pd.Series(all_hashtags).value_counts()
top_hashtag_n = int(len(hashtag_counts) * TOP_PERCENT)
popular_hashtag_set = set(hashtag_counts.head(top_hashtag_n).index)

# parse data
def parse_item_to_row(item, popular_music_set, popular_author_set, popular_hashtag_set):
    desc = item.get("desc", "")
    create_time = item.get("createTime", None)
    video = item.get("video", {})
    music = item.get("music", {})
    stats = item.get("stats", {})
    text_extra = item.get("textExtra", [])

    width = video.get("width")
    height = video.get("height")
    duration = video.get("duration")
    ratio = video.get("ratio")
    aspect_ratio = width / height if height else None

    hour_posted = None
    weekday_posted = None
    is_weekend = None
    if create_time:
        dt = datetime.utcfromtimestamp(create_time)
        hour_posted = dt.hour
        weekday_posted = dt.weekday()
        is_weekend = int(weekday_posted >= 5)

    music_title = music.get("title", "")
    music_author = music.get("authorName", "")

    hashtags = [
        tag.get("hashtagName", "").lower()
        for tag in text_extra
        if tag.get("type") == 1 and tag.get("hashtagName")
    ]
    num_trending_hashtags = sum(1 for tag in hashtags if tag in popular_hashtag_set)
    has_trending_hashtag = int(num_trending_hashtags > 0)

    return {
        # video
        "video_duration": duration,
        "video_width": width,
        "video_height": height,
        "aspect_ratio": aspect_ratio,
        "video_quality": ratio,

        # desc
        "desc_length": len(desc),
        "has_mention": int(any(tag.get("userUniqueId") for tag in text_extra)),

        # time
        "hour_posted": hour_posted,
        "weekday_posted": weekday_posted,
        "is_weekend": is_weekend,

        # music popularity
        "is_trending_music": int(music_title in popular_music_set),
        "is_popular_author": int(music_author in popular_author_set),

        # hashtag popularity
        "num_trending_hashtags": num_trending_hashtags,
        "has_trending_hashtag": has_trending_hashtag,

        # engagement
        "playCount": stats.get("playCount", 0),
        "diggCount": stats.get("diggCount", 0),
        "commentCount": stats.get("commentCount", 0),
        "shareCount": stats.get("shareCount", 0)
    }

# build dataset
dataset = pd.DataFrame([
    parse_item_to_row(item, popular_music_set, popular_author_set, popular_hashtag_set)
    for item in raw_data
])

# one-hot encode video_quality
dataset = pd.get_dummies(dataset, columns=["video_quality"], prefix="quality")

# build engagement_score
scaler = MinMaxScaler()
scaled_engagement = scaler.fit_transform(
    dataset[['playCount', 'diggCount', 'commentCount', 'shareCount']]
)
scaled_df = pd.DataFrame(scaled_engagement, columns=['play', 'digg', 'comment', 'share'])

dataset['engagement_score'] = (
    0.4 * scaled_df['play'] +
    0.3 * scaled_df['digg'] +
    0.2 * scaled_df['comment'] +
    0.1 * scaled_df['share']
)

# print correlation
all_feature_columns = [
    'video_duration', 'video_width', 'video_height', 'aspect_ratio',
    'desc_length', 'has_mention',
    'hour_posted', 'weekday_posted', 'is_weekend',
    'is_trending_music', 'is_popular_author',
    'num_trending_hashtags', 'has_trending_hashtag'
]

correlation = dataset[all_feature_columns + ['engagement_score']].corr()['engagement_score'].sort_values(ascending=False)

print("📊 Correlation with engagement_score:")
print(correlation)
