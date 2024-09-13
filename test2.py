import os
import django
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import matplotlib.pyplot as plt

# Set up Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

# Import your models
from trending.models import Track

# Load all track data
def load_data():
    tracks = Track.objects.all()
    data = []

    for track in tracks:
        data.append({
            'danceability': track.danceability,
            'energy': track.energy,
            'tempo': track.tempo,
            'valence': track.valence,
            'speechiness': track.speechiness,
            'acousticness': track.acousticness,
            'instrumentalness': track.instrumentalness,
            'liveness': track.liveness,
            'added_to_playlists_count': track.added_to_playlists_count,
            'popularity': track.popularity
        })

    return pd.DataFrame(data)

# Preprocess the data
def preprocess_data(df):
    df = df.dropna()  # Drop rows with missing values
    return df

# Feature importance analysis
def analyze_feature_importance(df):
    X = df.drop('popularity', axis=1)
    y = df['popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_

    gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbm.fit(X_train, y_train)
    gbm_importances = gbm.feature_importances_

    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'RandomForest_Importance': rf_importances,
        'GradientBoosting_Importance': gbm_importances
    }).sort_values(by='RandomForest_Importance', ascending=False)

    print(importance_df)

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    ax[0].barh(importance_df['Feature'], importance_df['RandomForest_Importance'])
    ax[0].set_title('Random Forest Feature Importance')
    ax[0].set_xlabel('Importance')

    ax[1].barh(importance_df['Feature'], importance_df['GradientBoosting_Importance'])
    ax[1].set_title('Gradient Boosting Feature Importance')
    ax[1].set_xlabel('Importance')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    analyze_feature_importance(df)
