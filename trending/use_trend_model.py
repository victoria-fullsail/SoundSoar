import joblib
import os 

def load_active_model(active_model):
    """
    Load the active model from disk based on the active version in the TrendModel table.
    
    :return: Loaded model, feature names
    :raises RuntimeError: If no active model is found
    """
    print('enter load active model')

    if not active_model:
        raise RuntimeError("No active model found in the database.")

    # Construct the file path using the version number
    model_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'trending/ml_models',
        f'{active_model.version_number}_model.pkl'
    )

    # Check if the model file exists
    if not os.path.exists(model_file_path):
        raise RuntimeError(f"Model file not found: {model_file_path}")

    # Load the model
    model, feature_names, imputer = joblib.load(model_file_path)

    print('exiting load active model')
    return model, feature_names, imputer

