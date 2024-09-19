import os
import sys
import django
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import pandas as pd
from data_preparations import load_data

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestClassifier(),
        'HistGradientBoosting': HistGradientBoostingClassifier()
    }

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
        }
    }

    best_model = None
    best_accuracy = 0
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model_candidate = grid_search.best_estimator_
        y_pred = best_model_candidate.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion.tolist(),
            'best_params': grid_search.best_params_
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_model_candidate

    return results, best_model

# Function to save a model to disk
def save_model(model, file_path):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

if __name__ == '__main__':
    results, best_model = train_and_evaluate_models()
    print("Results:", results)
    if best_model:
        model_file_path = os.path.join(BASE_DIR, 'trending/ml_models', 'best_model.pkl')
        save_model(best_model, model_file_path)
