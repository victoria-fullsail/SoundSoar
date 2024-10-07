from trend_model import train_and_evaluate_models, save_model
from trending.models import TrendModel

# Function to save a model with version number
def save_model_with_version(model, version_number, feature_names=None):
    model_file_path = f'trending/ml_models/{version_number}_model.pkl'
    save_model(model, model_file_path)

    # Optionally save feature names
    if feature_names:
        with open(f'trending/ml_models/{version_number}_feature_names.txt', 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")

    return model_file_path


# Function to update the active model in the database
def update_active_models():
    results, all_trained_models, best_model, feature_names = train_and_evaluate_models()
    
    if not best_model:
        print("No best model found.")
        return

    # Deactivate all models
    TrendModel.objects.update(is_active=False)

    # Find the best model from results
    best_model_name = max(results, key=lambda name: results[name]['accuracy'])
    best_model_results = results[best_model_name]

    # Create a new version in the database
    new_version = TrendModel.objects.create(
        model_type=best_model_name,
        accuracy=best_model_results['accuracy'],
        precision=best_model_results['classification_report']['weighted avg']['precision'],
        recall=best_model_results['classification_report']['weighted avg']['recall'],
        f1_score=best_model_results['classification_report']['weighted avg']['f1-score'],
        roc_auc=best_model_results.get('roc_auc', None),
        description=f'Best performing model: {best_model_name}',
        is_best=True
    )
    new_version.activate()

    # Save the model with version number
    model_file_path = save_model_with_version(best_model, new_version.version_number, feature_names) 
    print(f"Model version {new_version.version_number} saved at {model_file_path}")

    # Save other models
    for model_name, model in all_trained_models.items():
        if model_name != best_model_name:  # Exclude the best model
            model_results = results[model_name]
            new_version = TrendModel.objects.create(
                model_type=model_name,
                accuracy=model_results['accuracy'],
                precision=model_results['classification_report']['weighted avg']['precision'],
                recall=model_results['classification_report']['weighted avg']['recall'],
                f1_score=model_results['classification_report']['weighted avg']['f1-score'],
                roc_auc=model_results.get('roc_auc', None),
                description=f'Other model: {model_name}',
                is_best=False  # Not the best model
            )
            new_version.activate()

            # Save the model with version number
            model_file_path = save_model_with_version(model, new_version.version_number, feature_names)
            print(f"Model version {new_version.version_number} saved at {model_file_path}")


if __name__ == '__main__':
    update_active_models()
