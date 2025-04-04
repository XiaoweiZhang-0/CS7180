import pandas as pd

import streamlit as st
import pandas as pd
import numpy as np
import pickle

X = pd.read_csv("X_features.csv")
print(X.columns.tolist())

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define main features to input
st.title("TikTok Popularity Predictor")
st.write("Input video features below to predict popularity level:")

# User input form
duration = st.slider("Video Duration (seconds)", 0, 300, 30)
aspect_ratio = st.number_input("Aspect Ratio (height / width)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
verified = st.selectbox("Is the creator verified?", [0, 1])
hour = st.slider("Upload Hour (0–23)", 0, 23, 12)
weekday = st.selectbox("Upload Weekday (0=Mon, 6=Sun)", list(range(7)))

# Build input array for prediction
# All TF-IDF features default to 0
all_features = [
    'video.duration', '30x16', '42s', '4x4diesels', 'art', 'artist', 'ashleyvee', 'atv', 'bagged', 'brandon24v',
    'brandon24voffroad', 'burnout', 'car', 'cars', 'checkmeoutchallenge', 'chevy', 'clean', 'coldstart',
    'coronavirus', 'cummins', 'custom', 'diesel', 'dieselcheck', 'dieselgang', 'dieselpower', 'diesels',
    'dieselsmerica', 'dieseltruck', 'dieseltrucks', 'dieseltrux', 'dodge', 'dontflop', 'dontletthisflop', 'dually',
    'euro', 'exhaust', 'exotic', 'f250', 'f350', 'florida', 'floridacheck', 'for', 'forces', 'ford', 'forged',
    'foru', 'foryou', 'foryoupage', 'foyoupage', 'fullsend', 'fyp', 'goodbye2019', 'gotcaught', 'goviral',
    'hornblasters', 'houston', 'howidothings', 'howiwalk', 'jdm', 'lamborghini', 'lifted', 'liftedtruck',
    'liftedtruckcheck', 'liftedtrucks', 'liftedtrucksmatter', 'lowered', 'luxury', 'mechanic', 'minivlog',
    'pavementprincess', 'polaris', 'powerstroke', 'quarantine', 'racetruck', 'ram', 'red', 'rocklights', 'sema',
    'semafun', 'south', 'southerntruckz', 'speed', 'stance', 'supercar', 'thatswhatilike', 'tires', 'truck',
    'truckcheck', 'trucklife', 'truckporn', 'trucks', 'viral', 'w2step', 'weld', 'welding', 'wheels', 'work',
    'xyzbca', 'xyzcba', 'yeeyee', 'youtube', 'aspect_ratio', 'resolution', 'verified', 'hour', 'weekday', 'music_id'
]

# Create input row with default 0s
input_data = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)

# Update user-input fields
input_data['video.duration'] = duration
input_data['aspect_ratio'] = aspect_ratio
input_data['verified'] = verified
input_data['hour'] = hour
input_data['weekday'] = weekday

# Predict engagement score
if st.button("Predict Popularity"):
    # Drop features not seen during model training
    if 'resolution' in input_data.columns:
        input_data = input_data.drop(columns=['resolution'])

    prediction = model.predict(input_data)[0]

    # Convert score into categories
    if prediction >= 0.75:
        label = "🔥 1"
    elif prediction >= 0.5:
        label = "👍 2"
    elif prediction >= 0.25:
        label = "🙂 3"
    else:
        label = "😐 4"

    st.subheader("Prediction Result")
    st.write(f"Predicted engagement score: **{round(prediction, 4)}**")
    st.success(f"Popularity Level: {label}")
