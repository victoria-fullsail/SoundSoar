import os
import django

# Set up Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

# Import your models
from trending.models import TrackFeatures

def update_all_track_features():
    """
    Update all TrackFeatures instances in the database.
    """
    # Retrieve all TrackFeatures instances
    track_features_queryset = TrackFeatures.objects.all()

    # Loop through each instance and call update_features
    for feature_instance in track_features_queryset:
        print(f"Updating TrackFeatures ID: {feature_instance.id} - {feature_instance.track.name}")
        
        # Call update_features method
        feature_instance.update_features()

        # Print confirmation after update
        print(f"Updated TrackFeatures ID: {feature_instance.id}")

if __name__ == "__main__":
    update_all_track_features()
