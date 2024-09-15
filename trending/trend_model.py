import os
import sys
import django
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import pandas as pd
from data_preparations import load_data

# Set up Django environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def train_and_evaluate_models():
    """
    Train and evaluate models using the TrackFeatures data.
    Returns a dictionary with model names and their evaluation results.
    """
    # Load data
    data = load_data(model_name='TrackFeatures')

    # Define features and target
    X = data[['track__valence', 'track__tempo', 'track__speechiness', 'track__danceability', 'track__liveness', 'velocity', 'current_popularity', 'median_popularity', 'mean_popularity', 'std_popularity', 'retrieval_frequency']]
    y = data['trend']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and hyperparameters
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'HistGradientBoosting': {
            'model': HistGradientBoostingClassifier(),
            'params': {
                'max_iter': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            }
        }
    }

    results = {}
    best_model_name = None
    best_model = None
    best_accuracy = 0

    for name, info in models.items():
        print(f"Evaluating {name}...")

        # Hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(info['model'], info['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model_candidate = grid_search.best_estimator_
        y_pred = best_model_candidate.predict(X_test)

        # Collect evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion.tolist(),  # Convert numpy array to list for easier JSON serialization
            'best_params': grid_search.best_params_
        }
        
        print(f"{name} evaluation complete.\n")

        # Determine the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = best_model_candidate

    # Save the best model to disk
    if best_model:
        model_file_path = os.path.join(BASE_DIR, 'trending/ml_models', 'best_model.pkl')
        joblib.dump(best_model, model_file_path)
        print(f"Best model saved to {model_file_path}")

    return results, model_file_path


if __name__ == '__main__':
    results, model_file_path = train_and_evaluate_models()
    print("Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']}")
        print(f"  Best Params: {metrics['best_params']}")
        print(f"  Classification Report: {metrics['classification_report']}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
