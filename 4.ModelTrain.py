import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('tiktok_dataset_with_engagement_score.csv')

X = df[[
    'video.duration',
    'desc_len',
    'num_hashtags',
    'verified',
    'hour',
    'weekday',
    'is_original_sound',
    'stats.diggCount',
    'stats.commentCount',
    'stats.shareCount'
]]

y = df['engagement_score']

# split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# intialize model and train
#

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# predict y value 
y_pred = model.predict(X_test)

print("✅")