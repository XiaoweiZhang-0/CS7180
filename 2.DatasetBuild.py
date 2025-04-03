import json
import pandas as pd
import re
from collections import Counter

# Load and flatten JSON
with open('first_200000_entries.json', 'r') as f:
    data = json.load(f)

# filter out all secret videos
data = [entry for entry in data if (not entry.get('secret', False) and not entry.get('forFriend', False))]
print(f"✅ Videos after filtering (only public): {len(data)}")

df = pd.json_normalize(data, sep='.')

# Print available columns
print("\nAvailable columns:")
print(df.columns.tolist())

# Select features
df_selected = df[[
    'id',
    # Author features
    'author.verified',
    'author.uniqueId',
    'author.nickname',
    'author.signature',
    'author.relation',
    # Challenges features
    'challenges'
]].copy()

# Fill missing values with appropriate defaults
df_selected['author.verified'] = df_selected['author.verified'].fillna(False).astype(int)
df_selected['author.relation'] = df_selected['author.relation'].fillna(0)

# Extract hashtags using regex
def extract_hashtags(text):
    return re.findall(r'#\w+', str(text))

df_selected['hashtags'] = df_selected['desc'].apply(extract_hashtags)

# Flatten and count hashtags
all_hashtags = [tag.lower() for tags in df_selected['hashtags'] for tag in tags]
hashtag_counts = Counter(all_hashtags)

# Build a ranking dictionary {hashtag: rank}
sorted_hashtags = [tag for tag, _ in hashtag_counts.most_common()]
hashtag_rank = {tag: idx for idx, tag in enumerate(sorted_hashtags)}

# Assign rank of top hashtag in the post (lowest index = most popular)
def hashtag_freq_feature(tags):
    tags_lower = [tag.lower() for tag in tags]
    ranks = [hashtag_rank[tag] for tag in tags_lower if tag in hashtag_rank]
    return min(ranks) if ranks else 101  # 101 if no known hashtags present

df_selected['hashtag_freq_feature'] = df_selected['hashtags'].apply(hashtag_freq_feature)

print(df_selected.head())

# Save to CSV
df_selected.to_csv('tiktok_dataset.csv', index=False)
print("✅")
