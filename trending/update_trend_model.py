from trend_model import train_and_evaluate_models, save_model
from trending.models import TrendModel, FeatureImportance
from django.core.files import File

def save_model_with_version(model, version_number, imputer, feature_names, trend_model_obj):
    """Function to save a model with version number."""

    # Save model file
    model_file_path = f'trending/ml_models/{version_number}_model.pkl'
    save_model(model, feature_names, imputer, model_file_path)
    print(f"Model version {version_number} saved at {model_file_path}")

    pkl_filename = trend_model_obj.generate_filename('pkl')
    with open(model_file_path, 'rb') as f:
        trend_model_obj.model_file.save(pkl_filename, File(f), save=True)
    print(f"Pkl file uploaded and saved as {pkl_filename} in the model object.")
    

def save_csv_data_with_version(data, version_number, model_obj):
    """Save data to CSV and link to the model version in the database."""
    
    if not data.empty:
        csv_file_path = f'trending/ml_models/{version_number}_data.csv'
        data.to_csv(csv_file_path, index=False)
        
        print(f"Data for version {version_number} saved to CSV at {csv_file_path}.")

        # Open the file and save it to the model's csv_data field
        upload_filename = model_obj.generate_filename('csv')

        with open(csv_file_path, 'rb') as csv_file:
            model_obj.csv_data.save(upload_filename, File(csv_file), save=True)
            print(f"CSV file uploaded and saved as {upload_filename} in the model object.")
        

def save_feature_importances_to_version(trend_model_obj, feature_importances):
    """Save feature importances to the database."""

    for feature_name, importance in feature_importances.items():
        FeatureImportance.objects.create(
            trend_model=trend_model_obj,
            feature_name=feature_name,
            importance=importance
        )
    print(f"Feature importances saved for model {trend_model_obj.model_type}.")


def create_readme_file(trend_model_obj, feature_names):

    # Get relevant information
    model_type = trend_model_obj.model_type
    version_number = trend_model_obj.version_number
    best_params = trend_model_obj.best_parameters
    accuracy = trend_model_obj.accuracy
    precision = trend_model_obj.precision
    recall = trend_model_obj.recall
    f1_score = trend_model_obj.f1_score
    roc_auc = trend_model_obj.roc_auc
    confusion_matrix = trend_model_obj.confusion_matrix

    readme_content = (
        f"# Model Version: {version_number}\n\n"
        f"## Model Type: {model_type}\n\n"
        "## Best Parameters\n"
        "```\n"
        f"{best_params}\n"
        "```\n\n"
        "## Model Performance Metrics\n"
        f"- **Accuracy**: {accuracy}\n"
        f"- **Precision**: {precision}\n"
        f"- **Recall**: {recall}\n"
        f"- **F1 Score**: {f1_score}\n"
        f"- **ROC AUC**: {roc_auc}\n\n"
        "## Confusion Matrix\n"
        "```\n"
        f"{confusion_matrix}\n"
        "```\n\n"
        "## Feature List\n"
        "The following features were used in training the model:\n"
        "```\n"
        f"{chr(10).join(feature_names)}"
        "```\n\n"
        "## How to Use\n"
        "- Load the model and use the feature data to make predictions on target trend.\n"
    )

    readme_filename = trend_model_obj.generate_filename('txt')
    readme_file_path = f'trending/ml_models/{readme_filename}'

    # Save the README content to a text file temporarily
    with open(readme_file_path, 'w') as readme_file:
        readme_file.write(readme_content)
    print(f"README file saved at {readme_file_path}.")

    # Now save the README file to the TrendModel object
    with open(readme_file_path, 'rb') as f:
        trend_model_obj.readme_file.save(readme_filename, File(f), save=True)
    print(f"TXT file saved as {readme_filename} fort readme_file in TrendModel.")


def update_active_models():
    """Function to update the active model and also update the database."""

    # Call the train and evaulate model function to get all trend model information
    results, all_trained_models, best_model, imputer, data, feature_names, feature_importances = train_and_evaluate_models()

    # Deactivate all models
    TrendModel.objects.update(is_active=False)

    # Find the best model from results based on accuracy
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
        best_parameters=best_model_results['best_params'],
        confusion_matrix=str(best_model_results['confusion_matrix']),
        is_best=True
    )
    new_version.activate()

    # Save the model with version number
    save_model_with_version(best_model, new_version.version_number, imputer, feature_names, new_version) 

    # Save CSV of data
    save_csv_data_with_version(data, new_version.version_number, new_version)

    # Save Feature Importance
    save_feature_importances_to_version(new_version, feature_importances[best_model_name])

    # Generate Read Me File
    create_readme_file(new_version, feature_names)

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
                best_parameters=model_results['best_params'],  # Use correct results variable
                confusion_matrix=str(model_results['confusion_matrix']),
                is_best=False  # Not the best model
            )
            new_version.activate()

            # Save the model with version number
            save_model_with_version(model, new_version.version_number, imputer, feature_names, new_version)
        
            # Save CSV of data
            save_csv_data_with_version(data, new_version.version_number, new_version)

            # Save Feature Importance
            save_feature_importances_to_version(new_version, feature_importances[model_name])

            # Generate Read Me File
            create_readme_file(new_version, feature_names)


if __name__ == '__main__':
    update_active_models()
