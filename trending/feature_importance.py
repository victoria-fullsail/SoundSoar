import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from data_preparations import load_data

# Load your data
data = load_data(model_name='Track')
#X = data[['danceability', 'energy', 'tempo', 'valence', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']]
# These features were found to be the most important for RandomForest and GradientBoostingRegressor
X = data[['danceability', 'tempo', 'valence', 'speechiness', 'liveness']]
y = data['popularity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'LinearRegression': LinearRegression()
}

# Plot feature importance for models that support it and display coefficients for others
for name, model in models.items():
    model.fit(X_train, y_train)
    
    if hasattr(model, 'feature_importances_'):
        # Plot feature importance
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure()
        plt.title(f"Feature Importance: {name}")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()

        # Print feature importances
        print(f"Feature Importances for {name}:")
        for i in indices:
            print(f"{X.columns[i]}: {importances[i]}")
        print()
    
    elif hasattr(model, 'coef_'):
        # Display coefficients for linear models
        coefficients = model.coef_
        indices = np.argsort(np.abs(coefficients))[::-1]
        
        plt.figure()
        plt.title(f"Feature Coefficients: {name}")
        plt.bar(range(X.shape[1]), coefficients[indices], align="center")
        plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Coefficient")
        plt.tight_layout()
        plt.show()

    # Display cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-Validation Scores for {name}: {cv_scores}")
    print(f"Mean Cross-Validation Score for {name}: {np.mean(cv_scores)}\n")
