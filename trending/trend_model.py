import os
import sys
import django
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance 
from data_preparations import load_data


# Set up Django environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()


def save_model(model, feature_names, imputer, file_path):
    """Saves model to loacl disk using joblib."""
    try:
        joblib.dump((model, feature_names, imputer), file_path)
        print(f"Model saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def get_feature_importance(model, X_train, y_train):
    """Calculate feature importance for the given model."""

    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    
    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_[0])
    
    elif isinstance(model, (HistGradientBoostingClassifier, SVC, KNeighborsClassifier)):
        # Use permutation importance for applicable models
        perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
        return perm_importance.importances_mean
    
    elif isinstance(model, LinearDiscriminantAnalysis):
        return np.abs(model.coef_[0])
    
    else:
        print("Model does not support feature importance.")
        return None

def train_and_evaluate_models():
    """Train and Evaluate Models for TrackFeatures."""

    data = load_data(model_name='TrackFeatures')
    X = data[['track__valence', 'track__tempo', 'track__speechiness', 'track__danceability', 'track__liveness', 'velocity', 'current_popularity', 'median_popularity', 'mean_popularity', 'std_popularity', 'retrieval_frequency']]
    y = data['trend']

    # Save feature names
    feature_names = list(X.columns)   

    # Configure Imputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Define Scaler
    scaler = StandardScaler()

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Define the models to include
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1500, warm_start=True),
        'SVM': SVC(random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'ExtraTrees': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier()   
    }

    # Define the parameter grids for each model
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'HistGradientBoosting': {
            'max_iter': [100, 200, 300],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 50, 100, 500],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga', 'newton-cg']
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly']
        },
        'LDA': {
            'solver': ['svd', 'lsqr', 'eigen']
        },
        'ExtraTrees': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    }

    # Define other important variables
    best_model = None
    best_accuracy = 0
    results = {}
    all_trained_models = {}
    feature_importances = {}

    # Iterate over each model, apply GridSearchCV, and evaluate results
    for name, model in models.items():
        print(f"Training {name}...")

        # Check if the model needs scaling (LogisticRegression, SVM, KNN)
        if name in ['LogisticRegression', 'SVM', 'KNN']:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Use Grid Search to find the best parameters for the current model
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        # Get the best model found by GridSearchCV
        best_model_candidate = grid_search.best_estimator_

        # Make predictions with the best model on the test set
        y_pred = best_model_candidate.predict(X_test_scaled)

        # Calculate accuracy and generate a classification report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)

        # Store results for the current model
        results[name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': np.array2string(confusion, separator=', '), # Format as string
            'best_params': ", ".join([f"{key}: {value}" for key, value in grid_search.best_params_.items()])  # Format as string
        }

        # Save the trained model for later comparison
        all_trained_models[name] = best_model_candidate

        # Track the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_model_candidate

        # Feature Importance
        importance = get_feature_importance(best_model_candidate, X_train_scaled, y_train)

        if importance is not None:
            feature_importance_dict = {feature: importance[i] for i, feature in enumerate(feature_names)}
            feature_importances[name] = feature_importance_dict
        else:
            feature_importances[name] = None

    # Return all results, all models, and the best model
    return results, all_trained_models, best_model, imputer, data, feature_names, feature_importances
