import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance

# Function to load CSV data
def load_data_from_csv(csv_file):
    """Load dataset from a CSV file."""
    return pd.read_csv(csv_file)

# Function to train models and print results
def train_and_evaluate_models(csv_file):
    """Train and evaluate models from CSV input and print results."""
    data = load_data_from_csv(csv_file)
    
    # Adjust this based on the columns in your CSV
    X = data[['valence', 'tempo', 'speechiness', 'danceability', 'liveness', 'velocity', 'current_popularity']]
    y = data['trend_label']  # Assuming this is your target column
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess the data
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)
    
    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'HistGradient Boosting': HistGradientBoostingClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'LDA': LinearDiscriminantAnalysis()
    }
    
    # Training and evaluating models
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"\n{model_name} Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance (if applicable)
        if model_name in ['Random Forest', 'Extra Trees', 'HistGradient Boosting']:
            importances = model.feature_importances_
            print(f"Feature Importances for {model_name}:")
            for feature, importance in zip(X.columns, importances):
                print(f"{feature}: {importance:.4f}")
        elif model_name == 'SVM' or model_name == 'KNN':
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            print(f"Permutation Importances for {model_name}:")
            for feature, importance in zip(X.columns, perm_importance.importances_mean):
                print(f"{feature}: {importance:.4f}")

# Example of how to call the function
csv_file = 'path_to_your_csv_file.csv'
train_and_evaluate_models(csv_file)
