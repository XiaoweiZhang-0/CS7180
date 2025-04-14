import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import re
from sentence_transformers import SentenceTransformer
from sklearn.calibration import LabelEncoder

# Must be the first Streamlit command
st.set_page_config(
    page_title="TikTok Video Popularity Prediction",
    page_icon="📱",
    layout="wide"
)

# Define functions
@st.cache_resource
def load_model():
    """Load the trained model from pickle file"""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_bert_model():
    """Load the BERT model for text embeddings"""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def remove_hashtags_and_mentions(text):
    """Remove hashtags and mentions from text for BERT processing"""
    text = re.sub(r"#[\w-]+", "", text)  # Remove hashtags
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    return text

def extract_hashtags(text):
    """Extract all hashtags from text"""
    return re.findall(r'#(\w+)', text)

def extract_mentions(text):
    """Extract all mentions from text"""
    return re.findall(r'@(\w+)', text)

def get_bert_embedding(text, bert_model):
    """Get BERT embedding for a given text"""
    processed_text = remove_hashtags_and_mentions(text)
    embedding = bert_model.encode(processed_text)
    return embedding

def categorize_prediction(pred_idx):
    """Map prediction index to category details"""
    categories = {
        0: {"name": "Low Engagement", "emoji": "😐", "color": "#C7C7C7"},
        1: {"name": "Average Engagement", "emoji": "🙂", "color": "#DAF7A6"},
        2: {"name": "Popular", "emoji": "👍", "color": "#FFC300"},
        3: {"name": "Highly Popular", "emoji": "🔥", "color": "#FF5733"}
    }
    return categories.get(pred_idx, {"name": "Unknown", "emoji": "❓", "color": "#CCCCCC"})

# Title and description
st.title("🎬 TikTok Video Popularity Predictor")
st.markdown("""
This application predicts how popular your TikTok video will be based on various features.
Fill in the details below to get a prediction!
""")

# Try to load models
try:
    model = load_model()
    bert_model = load_bert_model()
    models_loaded = True
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.warning("⚠️ Running in demonstration mode. Predictions will be random.")
    models_loaded = False

# Create two-column layout
col1, col2 = st.columns([3, 2])

with col1:
    # Video content section
    st.header("📝 Video Content")
    
    # Video description
    video_description = st.text_area(
        "Video Description (include hashtags with # and mentions with @)", 
        value="Check out my new dance! #fyp #dance #viral @friendname",
        height=100,
        help="Enter your video description exactly as you would on TikTok, including hashtags and mentions"
    )
    
    # Automatically extract and display hashtags and mentions
    hashtags = extract_hashtags(video_description)
    mentions = extract_mentions(video_description)
    
    # Calculate description length
    desc_length = len(video_description)
    
    # Display extracted information
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"📏 Description Length: {desc_length} characters")
        st.write(f"🔖 Hashtags Found: {len(hashtags)}")
        if hashtags:
            st.write("  " + " ".join([f"#{tag}" for tag in hashtags]))
    
    with col_b:
        st.write(f"👥 Mentions Found: {len(mentions)}")
        if mentions:
            st.write("  " + " ".join([f"@{mention}" for mention in mentions]))
    
    # Video duration
    video_duration = st.slider(
        "Video Duration (seconds)", 
        min_value=1, 
        max_value=60, 
        value=15,
        help="Duration of your video in seconds"
    )
    
    # Posting time
    hour_posted = st.slider(
        "Hour of Posting (0-23)", 
        min_value=0, 
        max_value=23, 
        value=18,
        help="Hour of the day when you plan to post the video (0 = midnight, 12 = noon, 18 = 6 PM)"
    )
    
    # Challenges section
    st.header("🎯 Video Challenges")
    st.markdown("Enter any challenges you're participating in with this video")
    
    # Dynamic challenge inputs
    num_challenge_inputs = st.number_input("Number of Challenges", min_value=0, max_value=10, value=2)
    
    challenges = []
    challenge_details = []
    
    for i in range(int(num_challenge_inputs)):
        cols = st.columns([3, 1, 1])
        with cols[0]:
            challenge_name = st.text_input(f"Challenge {i+1}", value=f"Challenge{i+1}", key=f"ch{i}")
        with cols[1]:
            has_desc = st.checkbox(f"Has Description", value=True, key=f"desc{i}")
        with cols[2]:
            has_thumb = st.checkbox(f"Has Thumbnail", value=True, key=f"thumb{i}")
        
        if challenge_name:
            challenges.append(challenge_name)
            challenge_details.append({"name": challenge_name, "has_desc": has_desc, "has_thumb": has_thumb})
    
    # Calculate challenge completeness (percentage of challenges with both desc and thumb)
    complete_challenges = sum(1 for c in challenge_details if c["has_desc"] and c["has_thumb"])
    challenge_completeness = complete_challenges / len(challenge_details) if challenge_details else 0
    
    # Average challenge description length (using a reasonable default since we don't ask for description content)
    avg_challenge_desc_length = 50

with col2:
    # Creator information
    st.header("👤 Creator Information")
    
    # Whether the creator is verified
    author_verified = st.checkbox("Creator Has Verified Account", value=False)
    
    # Creator bio/signature
    author_signature = st.text_area(
        "Creator Bio/Signature", 
        value="Dancing and having fun! Follow for more content!",
        height=80,
        help="Enter the bio text that appears on your TikTok profile"
    )
    author_signature_len = len(author_signature)
    
    # Creator username
    author_nickname = st.text_input(
        "Creator Username", 
        value="dancerXYZ",
        help="Your TikTok username"
    )
    author_name_len = len(author_nickname)
    
    st.write(f"📏 Bio Length: {author_signature_len} characters")
    
    # Prediction button
    st.header("🔮 Get Prediction")
    predict_button = st.button("Predict Video Popularity", type="primary", use_container_width=True)

# Prediction section
if predict_button:
    with st.spinner("Analyzing your video..."):
        # Add slight delay for better UX
        time.sleep(1)
        
        if models_loaded:
            try:
                # Create feature input - numeric features
                features = {
                    "desc_length": desc_length,
                    "has_trending_hashtag": 1 if hashtags else 0,  # Assume at least one hashtag is trending if any exist
                    "num_trending_hashtags": len(hashtags),
                    "is_trending_music": 1,  # Simplify, assume using trending music
                    "has_mention": 1 if mentions else 0,
                    "hour_posted": hour_posted,
                    "video.duration": video_duration,
                    "author.verified": int(author_verified),
                    "author_signature_len": author_signature_len,
                    "author_name_len": author_name_len,
                    "avg_challenge_desc_length": avg_challenge_desc_length,
                    "num_challenges": len(challenges),
                    "challenge_completeness": challenge_completeness
                }
                
                # Convert to appropriate format
                X_numeric = np.array([list(features.values())])
                
                # Get BERT embeddings for text fields
                desc_embedding = get_bert_embedding(video_description, bert_model)
                author_sig_embedding = get_bert_embedding(author_signature, bert_model)
                author_name_embedding = get_bert_embedding(author_nickname, bert_model)
                challenges_text = " ".join(challenges)
                challenges_embedding = get_bert_embedding(challenges_text, bert_model)
                
                # Combine all embeddings
                text_embeddings = np.hstack([
                    desc_embedding,
                    author_sig_embedding,
                    author_name_embedding,
                    challenges_embedding
                ])
                
                # Combine numeric features with text embeddings
                X_combined = np.hstack([X_numeric, text_embeddings.reshape(1, -1)])
                
                # Make prediction
                prediction = model.predict(X_combined)[0]
                prediction_proba = model.predict_proba(X_combined)[0]
                
                # Get category details
                category_info = categorize_prediction(prediction)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.error("Using random prediction for demonstration")
                # Fallback for demo mode
                prediction = np.random.randint(0, 4)
                prediction_proba = np.random.random(4)
                prediction_proba = prediction_proba / prediction_proba.sum()  # Normalize to sum to 1
                category_info = categorize_prediction(prediction)
        else:
            # Demo mode
            prediction = np.random.randint(0, 4)
            prediction_proba = np.random.random(4)
            prediction_proba = prediction_proba / prediction_proba.sum()  # Normalize to sum to 1
            category_info = categorize_prediction(prediction)

        # Display results
        st.header("🎯 Prediction Results")
        
        # Display category with colored box
        st.markdown(f"""
        <div style='background-color:{category_info["color"]}; padding:20px; border-radius:10px; text-align:center;'>
            <h2 style='margin:0; color:white;'>Predicted Popularity: {category_info["name"]} {category_info["emoji"]}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations section
        st.subheader("💡 Recommendations to Improve")
        
        if prediction == 3:  # Highly Popular
            st.success("🎉 Your video has excellent potential! Key strengths:")
            st.markdown("- ✅ Strong use of hashtags")
            st.markdown("- ✅ Good posting time and video duration")
            st.markdown("- ✅ Effective challenge participation")
            st.markdown("- ✅ Well-crafted description and creator profile")
            
        elif prediction == 2:  # Popular
            st.info("👍 Your video has good potential! Consider these improvements:")
            
            if len(hashtags) < 5:
                st.markdown("- ➕ Add more hashtags (aim for 5+)")
            
            if hour_posted < 17 or hour_posted > 22:
                st.markdown("- ⏰ Consider posting between 5PM-10PM for better engagement")
                
            if not author_verified:
                st.markdown("- 👤 Work towards getting a verified account for higher credibility")
                
            if challenge_completeness < 0.8:
                st.markdown("- 🎯 Complete challenges more thoroughly with both descriptions and thumbnails")
                
        elif prediction == 1:  # Average
            st.warning("⚠️ Your video has moderate potential. Try these changes:")
            
            if len(hashtags) < 3:
                st.markdown("- ❗ Significantly increase hashtags (aim for 7+)")
            
            if hour_posted < 17 or hour_posted > 22:
                st.markdown("- ❗ Post only during peak hours (5PM-10PM)")
                
            if len(challenges) < 2:
                st.markdown("- 🎯 Participate in more popular challenges")
                
            if video_duration > 30 or video_duration < 10:
                st.markdown("- ⏱️ Aim for 15-30 second videos for optimal engagement")
                
        else:  # Low engagement
            st.error("⚠️ Your video might struggle to gain traction. Consider these major changes:")
            
            st.markdown("- 🔄 Use at least 7-10 hashtags")
            st.markdown("- 🔄 Post only during peak hours (7-10 PM)")
            st.markdown("- 🔄 Participate in multiple viral challenges")
            st.markdown("- 🔄 Keep videos concise (15-20 seconds)")
            st.markdown("- 🔄 Improve your creator profile and description")
            if not mentions:
                st.markdown("- 🔄 Consider mentioning other creators for wider reach")

# Footer
st.markdown("---")
st.markdown("TikTok Video Popularity Prediction | Powered by Machine Learning & BERT")
st.markdown("---")
st.caption("Built with Streamlit, BERT, and XGBoost for CS7180 Project")