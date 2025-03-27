import json
import pandas as pd

with open('subset.json', 'r') as f:
    data = json.load(f)

# filter out all secret videos
data = [entry for entry in data if (not entry.get('secret', False) and not entry.get('forFriend', False))]
print(f"✅ Videos after filtering (only public): {len(data)}")

df = pd.json_normalize(data, sep='.')

# if more features are  needed, could be added here
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
]]

df_selected = df_selected.copy()

df_selected['createTime'] = pd.to_datetime(df_selected['createTime'], unit='s')

df_selected.to_csv('tiktok_dataset.csv', index=False)
print("✅")