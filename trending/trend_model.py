from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os
import sys
import django
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
import pandas as pd
from data_preparations import load_data
from sklearn.preprocessing import StandardScaler


# Set up Django environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()


# Function to train models and evaluate them
def train_and_evaluate_models():
    data = load_data(model_name='TrackFeatures')
    X = data[['track__valence', 'track__tempo', 'track__speechiness', 'track__danceability', 'track__liveness', 'velocity', 'current_popularity', 'median_popularity', 'mean_popularity', 'std_popularity', 'retrieval_frequency']]
    y = data['trend']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Save feature names
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Define the models to include
    models = {
        'RandomForest': RandomForestClassifier(),
        'HistGradientBoosting': HistGradientBoostingClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC()
    }

    # Define the parameter grids for each model
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'HistGradientBoosting': {
            'max_iter': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly']
        },
    }

    best_model = None
    best_accuracy = 0
    results = {}
    all_trained_models = {}

    scaler = StandardScaler()

    # Iterate over each model, apply GridSearchCV, and evaluate results
    for name, model in models.items():
        print(f"Training {name}...")

        # Check if the model needs scaling (LogisticRegression, SVM)
        if name in ['LogisticRegression', 'SVM']:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        # Best model from GridSearchCV for current algorithm
        best_model_candidate = grid_search.best_estimator_
        y_pred = best_model_candidate.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)

        # Save the results for the current model
        results[name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion.tolist(),
            'best_params': grid_search.best_params_
        }

        # Save the trained model for later comparison
        all_trained_models[name] = best_model_candidate

        # Track the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_model_candidate

    # Return all results, all models, and the best model
    return results, all_trained_models, best_model, feature_names


