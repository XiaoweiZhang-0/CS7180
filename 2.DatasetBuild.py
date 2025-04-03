import json
import pandas as pd
import re
from collections import Counter
import numpy as np

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

# 3. Combined Features
df_selected['author_engagement_score'] = (
    df_selected['author.verified'] * 2 +
    df_selected['author_signature_len'] / 100 +  # Normalized by dividing by 100
    df_selected['author_name_len'] / 50  # Normalized by dividing by 50
)

df_selected['challenge_engagement_score'] = (
    df_selected['num_challenges'] +
    df_selected['challenges_with_desc'] * 1.5 +  # Weight more for challenges with descriptions
    df_selected['challenges_with_thumb'] * 1.2 +  # Weight more for challenges with thumbnails
    df_selected['challenge_completeness'] * 2  # Weight more for complete challenges
)

# Fill any remaining missing values with appropriate defaults
numeric_columns = df_selected.select_dtypes(include=[np.number]).columns
df_selected[numeric_columns] = df_selected[numeric_columns].fillna(df_selected[numeric_columns].median())

print("\nSelected features:")
print(df_selected.columns.tolist())

# Save to CSV
df_selected.to_csv('tiktok_dataset.csv', index=False)
print("✅")
