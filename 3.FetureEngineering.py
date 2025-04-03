import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('tiktok_dataset.csv')

print("Available columns:")
print(df.columns.tolist())

# Prepare features for analysis
features = [
    # Primary Author Features
    'author.verified',
    'author_signature_len',
    'author_name_len',
    # Challenge Features
    'num_challenges',
    'challenges_with_desc',
    'challenges_with_thumb',
    'avg_challenge_desc_length',
    'challenge_completeness',
    # Combined Scores
    'author_engagement_score',
    'challenge_engagement_score'
]

X = df[features]

# Fill missing values with median
X = X.fillna(X.median())

# 1. Correlation Analysis
correlation_matrix = X.corr()
print("\nFeature Correlations:")
print(correlation_matrix)

# 2. Mutual Information Analysis between features and challenge_engagement_score
target = X['challenge_engagement_score']
features_for_mi = [f for f in features if f != 'challenge_engagement_score']
X_for_mi = X[features_for_mi]

mi_scores = mutual_info_regression(X_for_mi, target)
mi_series = pd.Series(mi_scores, index=features_for_mi)
print("\nMutual Information Scores (with challenge_engagement_score):")
print(mi_series.sort_values(ascending=False))

# Create visualizations
plt.figure(figsize=(20, 15))

# Correlation Heatmap
plt.subplot(2, 1, 1)
plt.title('Feature Correlation Heatmap')

# Mutual Information Bar Plot
plt.subplot(2, 1, 2)
mi_series.sort_values().plot(kind='bar')
plt.title('Mutual Information Scores (with challenge_engagement_score)')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('feature_analysis.png')
plt.close()

# Print top correlated feature pairs
print("\nTop Correlated Feature Pairs:")
correlations = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        corr = abs(correlation_matrix.iloc[i, j])
        correlations.append((features[i], features[j], corr))

correlations.sort(key=lambda x: abs(x[2]), reverse=True)
for f1, f2, corr in correlations[:10]:
    print(f"{f1} - {f2}: {corr:.3f}")

# Print top features by mutual information
print("\nTop Features by Mutual Information (with challenge_engagement_score):")
print(mi_series.nlargest(5))

# Calculate feature importance based on average absolute correlation
feature_importance = abs(correlation_matrix).mean().sort_values(ascending=False)
print("\nFeature Importance (Based on Average Absolute Correlation):")
print(feature_importance.head()) 