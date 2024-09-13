import os
import django
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set up Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

# Import your models
from trending.models import TrackFeatures

# Load all track features
def load_data():
    track_features = TrackFeatures.objects.all()
    data = []

    for feature in track_features:
        data.append({
            'danceability': feature.danceability,
            'energy': feature.energy,
            'tempo': feature.tempo,
            'current_popularity': feature.current_popularity,
            'velocity': feature.velocity,
            'retrieval_frequency': feature.retrieval_frequency,
            'trend': feature.trend
        })

    return pd.DataFrame(data)

# Preprocess the data
def preprocess_data(df):
    df['retrieval_frequency'] = df['retrieval_frequency'].map({'high': 2, 'medium': 1, 'low': 0})
    df['trend'] = df['trend'].map({'up': 1, 'down': -1, 'stable': 0})
    df = df.dropna()
    return df

# Feature importance analysis
def analyze_feature_importance(df):
    X = df.drop('current_popularity', axis=1)
    y = df['current_popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_

    gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
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
