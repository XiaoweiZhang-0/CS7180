import json
import pandas as pd
import re

# load and flatten JSON
with open('subset_10000.json', 'r') as f:
    raw_data = json.load(f)

# filter out all secret videos
filtered = [entry for entry in raw_data if not entry.get("secret", False) and not entry.get("forFriend", False)]
print(f"✅ Public videos retained: {len(filtered)}")

# parse and extract each item
def extract_raw_fields(item):
    desc = item.get("desc", "")
    text_extra = item.get("textExtra", [])
    hashtags = re.findall(r"#\w+", desc)

    return {
        # id
        "id": item.get("id"),

        # for features of video
        "desc": desc,
        "hashtags": hashtags,
        "textExtra": text_extra,
        "createTime": item.get("createTime"),
        "video_duration": item.get("video", {}).get("duration"),

        # for features of music
        "music_title": item.get("music", {}).get("title"),

        # for egagment score
        "playCount": item.get("stats", {}).get("playCount", 0),
        "diggCount": item.get("stats", {}).get("diggCount", 0),
        "commentCount": item.get("stats", {}).get("commentCount", 0),
        "shareCount": item.get("stats", {}).get("shareCount", 0)
    }

# Build dataframe
df_raw = pd.DataFrame([extract_raw_fields(entry) for entry in filtered])

# Save clean raw dataset
df_raw.to_csv("tiktok_dataset.csv", index=False)
print("✅ Raw dataset saved as tiktok_dataset.csv")