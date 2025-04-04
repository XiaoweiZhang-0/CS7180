import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === Load model ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# === Define label decoder ===
label_decoder = {
    0: "low",
    1: "average",
    2: "popular",
    3: "highly popular"
}

st.title("🎬 Predict TikTok Video Popularity Level")
st.markdown("Input video metadata to predict how popular it might be.")

# === User Inputs ===
desc_length = st.slider("Description Length (characters)", 0, 300, 100)
has_trending_hashtag = st.selectbox("Has Trending Hashtag", [0, 1])
num_trending_hashtags = st.slider("Number of Trending Hashtags", 0, 10, 1)
is_trending_music = st.selectbox("Is Using Trending Music", [0, 1])
has_mention = st.selectbox("Mentions Another User", [0, 1])
hour_posted = st.slider("Hour of Posting (0–23)", 0, 23, 12)
video_duration = st.slider("Video Duration (seconds)", 0, 200, 30)
author_verified = st.selectbox("Author Verified", [0, 1])
author_signature_len = st.slider("Author Signature Length", 0, 200, 50)
author_name_len = st.slider("Author Username Length", 0, 100, 10)
avg_challenge_desc_length = st.slider("Average Challenge Description Length", 0, 300, 50)
num_challenges = st.slider("Number of Challenges", 0, 100, 10)
challenge_completeness = st.slider("Challenge Completeness (0.0 - 1.0)", 0.0, 1.0, 0.5, step=0.01)

# === Create input DataFrame ===
input_df = pd.DataFrame([[
    desc_length,
    has_trending_hashtag,
    num_trending_hashtags,
    is_trending_music,
    has_mention,
    hour_posted,
    video_duration,
    author_verified,
    author_signature_len,
    author_name_len,
    avg_challenge_desc_length,
    num_challenges,
    challenge_completeness
]], columns=[
    "desc_length",
    "has_trending_hashtag",
    "num_trending_hashtags",
    "is_trending_music",
    "has_mention",
    "hour_posted",
    "video.duration",
    "author.verified",
    "author_signature_len",
    "author_name_len",
    "avg_challenge_desc_length",
    "num_challenges",
    "challenge_completeness"
])

# === Predict and show result ===
if st.button("Predict Popularity Level"):
    pred_label_index = model.predict(input_df)[0]
    pred_label = label_decoder.get(pred_label_index, "Unknown")

    emoji_map = {
        "low": "😐",
        "average": "🙂",
        "popular": "👍",
        "highly popular": "🔥"
    }

    st.subheader("📊 Prediction Result")
    st.metric("Predicted Category", f"{pred_label.upper()} {emoji_map.get(pred_label, '')}")
