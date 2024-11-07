import os
import django
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

# Set up Django environment (adjust the path as necessary)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from trending.models import TrackFeatures

# Query all track features
tracks = TrackFeatures.objects.select_related('track').all()

# Prepare data dictionary for each feature
data = {
    'Current Popularity': [],
    'Danceability': [],
    'Energy': [],
    'Tempo': [],
    'Valence': [],
    'Speechiness': [],
    'Acousticness': [],
    'Instrumentalness': [],
    'Liveness': [],
    'Velocity': [],
    'Median Popularity': [],
    'Mean Popularity': [],
    'STD Popularity': [],
    'Retrieval Frequency': [],
    'Trend': []
}

# Populate the data dictionary with track features
for track in tracks:
    data['Current Popularity'].append(track.current_popularity)
    data['Danceability'].append(track.track.danceability)
    data['Energy'].append(track.track.energy)
    data['Tempo'].append(track.track.tempo)
    data['Valence'].append(track.track.valence)
    data['Speechiness'].append(track.track.speechiness)
    data['Acousticness'].append(track.track.acousticness)
    data['Instrumentalness'].append(track.track.instrumentalness)
    data['Liveness'].append(track.track.liveness)
    data['Velocity'].append(track.velocity)
    data['Median Popularity'].append(track.median_popularity)
    data['Mean Popularity'].append(track.mean_popularity)
    data['STD Popularity'].append(track.std_popularity)
    data['Retrieval Frequency'].append(track.retrieval_frequency)
    data['Trend'].append(track.trend)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data)

# Replace None with NaN for easier handling of missing values
df = df.replace({None: np.nan})

# Encode categorical variables for 'Retrieval Frequency' and 'Trend'
df['Retrieval Frequency'] = df['Retrieval Frequency'].map({
    'high': 2, 'medium': 1, 'low': 0
})

df['Trend'] = df['Trend'].map({
    'up': 1, 'down': -1, 'stable': 0
})

# Prepare features (X) and target (y)
X = df.drop(columns=['Trend'])  # Features
y = df['Trend']  # Target

# Handle missing values (impute)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, _, y_train, _ = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scale the features (important for models like Logistic Regression, SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the Random Forest model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)  # Use imputed features for RandomForest

# Initialize and train the HistGradientBoosting model
hgb = HistGradientBoostingClassifier(random_state=42)
hgb.fit(X_train, y_train)  # Use imputed features for HistGradientBoosting

# Get feature importances for Random Forest
rf_importances = rf.feature_importances_

# Calculate permutation importance for HistGradientBoosting
hgb_perm_importance = permutation_importance(hgb, X_train, y_train, n_repeats=10, random_state=42)
hgb_importances = hgb_perm_importance.importances_mean  # Mean importance from permutation results

# Create DataFrames to store feature importances
rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)

hgb_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': hgb_importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importance for Random Forest
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_importance_df)
plt.title('Feature Importance (Random Forest)')
plt.show()

# Plot feature importance for HistGradientBoosting (Permutation Importance)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=hgb_importance_df)
plt.title('Feature Importance (HistGradientBoosting - Permutation Importance)')
plt.show()

# Optionally, display the sorted feature importances
print("Random Forest Feature Importances:")
print(rf_importance_df)
print("\nHistGradientBoosting Feature Importances:")
print(hgb_importance_df)
