import os
import sys
import django
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance  # For permutation importance
from data_preparations import load_data

# Set up Django environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def calculate_feature_importance(X, y):
    """Train models and calculate feature importance."""
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, _, y_train, _ = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define the models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=500),
        'SVM': SVC(random_state=42, probability=True, C=1.0, kernel='linear')
    }

    # Initialize a DataFrame to store feature importance results
    feature_importance_results = {}

    # Train each model and calculate feature importance
    for name, model in models.items():
        print(f"Training {name}...")

        # Use scaled features for SVM and Logistic Regression
        if name in ['LogisticRegression', 'SVM']:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)  # Tree-based models can use unscaled features

        # Calculate feature importance
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For logistic regression
            importance = np.abs(model.coef_[0])
        elif name == 'HistGradientBoosting':
            # Using permutation importance for HistGradientBoosting
            results = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
            importance = results.importances_mean
        elif name == 'SVM':
            # Using permutation importance for SVM
            results = permutation_importance(model, X_train_scaled, y_train, n_repeats=10, random_state=42)
            importance = results.importances_mean
        else:
            print(f"Model {name} does not support feature importance.")
            continue

        # Store feature importance
        feature_importance_results[name] = importance

    return feature_importance_results


import matplotlib.pyplot as plt

def plot_feature_importance(feature_importance_results, feature_names):
    for name, importance in feature_importance_results.items():
        plt.figure(figsize=(12, 6))
        indices = np.argsort(importance)[::-1]  # Sort by importance
        plt.title(f'Feature Importance for {name}')
        plt.bar(range(len(importance)), importance[indices], align='center')
        plt.xticks(range(len(importance)), np.array(feature_names)[indices], rotation=90)
        plt.xlim([-1, len(importance)])
        plt.ylabel('Importance')
        plt.xlabel('Features')
        plt.tight_layout()
        plt.show()




def main():
    # Load the data
    data = load_data(model_name='TrackFeatures')
    X = data[['track__valence', 'track__tempo', 'track__speechiness', 'track__danceability', 
               'track__liveness', 'velocity', 'current_popularity', 'median_popularity', 
               'mean_popularity', 'std_popularity', 'retrieval_frequency']]
    y = data['trend']

    feature_importance_results = calculate_feature_importance(X, y)

    # Print the feature importances
    for model_name, importances in feature_importance_results.items():
        print(f"\nFeature Importances for {model_name}:")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print(importance_df)

    plot_feature_importance(feature_importance_results, X.columns)


if __name__ == "__main__":
    main()
