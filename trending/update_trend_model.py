import os
import sys
import django
from data_preparations import load_data
from trend_model import train_and_evaluate_models
from trending.models import TrendModel
import joblib

# Set up Django environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def save_model_to_disk(model, version_number):
    """
    Save the model to disk with the specified version number.
    
    :param model: Trained model to save
    :param version_number: Version number to include in the model filename
    """
    model_file_path = f'{version_number}_model.pkl'
    joblib.dump(model, model_file_path)

def update_active_model(results):
    """
    Update the active model version in the database based on evaluation results.
    
    :param results: Dictionary containing model evaluation results
    """
    # Deactivate all models
    TrendModel.objects.update(is_active=False)
    
    # Determine the best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model_results = results[best_model_name]

    # Create new model version entry
    new_version = TrendModel.objects.create(
        model_type=best_model_name,
        accuracy=best_model_results['accuracy'],
        precision=best_model_results['classification_report']['weighted avg']['precision'],
        recall=best_model_results['classification_report']['weighted avg']['recall'],
        f1_score=best_model_results['classification_report']['weighted avg']['f1-score'],
        roc_auc=best_model_results.get('roc_auc', None),
        description=f'Best performing model: {best_model_name}',
    )
    new_version.activate()

    # Load the best model from disk
    model_file_path = f'{new_version.version_number}_model.pkl'
    best_model = joblib.load(model_file_path)
    
    # Save the model to disk with version number
    save_model_to_disk(best_model, new_version.version_number)

if __name__ == "__main__":
    # Train and evaluate models
    results = train_and_evaluate_models()
    print(results)
    # Update active model and save the best model to disk
    update_active_model(results)
