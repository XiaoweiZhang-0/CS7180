import json
import pandas as pd
import re
from collections import Counter

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Load and flatten JSON
with open('subset.json', 'r') as f:
    data = json.load(f)

# filter out all secret videos
data = [entry for entry in data if (not entry.get('secret', False) and not entry.get('forFriend', False))]
print(f"✅ Videos after filtering (only public): {len(data)}")

df = pd.json_normalize(data, sep='.')

# Select features
df_selected = df[[
    'id',
    'desc',
    'createTime',
    'video.duration',
    'author.verified',
    'music.id',
    'video.width',
    'video.height',
    'video.ratio',
    'stats.playCount',
    'stats.diggCount',
    'stats.commentCount',
    'stats.shareCount'
]].copy()

# Convert timestamp
df_selected['createTime'] = pd.to_datetime(df_selected['createTime'], unit='s')

# Extract hashtags using regex
def extract_hashtags(text):
    return re.findall(r'#\w+', str(text))

df_selected['hashtags'] = df_selected['desc'].apply(extract_hashtags)


# Convert list of hashtags into space-separated strings
df_selected['hashtags_str'] = df_selected['hashtags'].apply(lambda x: ' '.join(x))

# Apply TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=100)  # Limit to top 100 important hashtags
tfidf_matrix = vectorizer.fit_transform(df_selected['hashtags_str']).toarray()


# Convert to DataFrame
hashtag_features = pd.DataFrame(tfidf_matrix, columns=vectorizer.get_feature_names_out())

# Merge with original DataFrame
df_selected = df_selected.drop(columns=['hashtags_str'])  # Drop temp string column
df_selected = pd.concat([df_selected, hashtag_features], axis=1)


# # Flatten and count hashtags
# # all_hashtags = [tag.lower() for tags in df_selected['hashtags'] for tag in tags]
# encoder = OneHotEncoder()
# mlb = MultiLabelBinarizer()
# # Transform hashtags
# transformed = mlb.fit_transform(df_selected['hashtags'])

# # Convert back to DataFrame with proper column names
# df_transformed = pd.DataFrame(transformed, columns=mlb.classes_)

# # Merge with original DataFrame if needed
# df_selected = df_selected.drop(columns=['hashtags']).join(df_transformed)

# # hashtag_counts = Counter(all_hashtags)


# # ## Summing up frequency of hashtags
# # num_of_counts = sum(hashtag_counts.values())

# # ## find the n most common hashtags such that the total frequency is at least 90% of the total counts
# # n_common = 0
# # cumulative_count = 0
# # for tag, count in hashtag_counts.most_common():
# #     cumulative_count += count
# #     n_common += 1
# #     if cumulative_count / num_of_counts >= 0.9:
# #         break
# # print(f"✅ Number of hashtags to keep for 90% frequency: {n_common}")


# # # Build a ranking dictionary {hashtag: rank}
# # sorted_hashtags = [tag for tag, _ in hashtag_counts.most_common(n_common)]
# # hashtag_rank = {tag: idx for idx, tag in enumerate(sorted_hashtags)}

# # # print(sorted_hashtags)


# # # Assign rank of top hashtag in the post (lowest index = most popular)
# # def hashtag_freq_feature(tags):
# #     tags_lower = [tag.lower() for tag in tags]
# #     ranks = [hashtag_rank[tag] for tag in tags_lower if tag in hashtag_rank]
# #     return ranks if ranks else -1  # -1 if no known hashtags present

# # df_selected['hashtag_freq_feature'] = df_selected['hashtags'].apply(hashtag_freq_feature)

# # print(df_selected.head())

# Save to CSV
df_selected.to_csv('tiktok_dataset.csv', index=False)
print("✅")
