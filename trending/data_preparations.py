import pandas as pd
import os
import sys
import django
from django.core.exceptions import ObjectDoesNotExist

# Set up the project base directory and Django settings module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
from trending.models import Track, TrackFeatures
import numpy as np

def load_data(model_name):
    """
    Load all data from the specified Django model and return it as a DataFrame.
    
    :param model_name: str, either 'Track' or 'TrackFeatures'
    :return: pandas DataFrame
    """
    if model_name == 'Track':
        queryset = Track.objects.all().values()
        df_track = pd.DataFrame.from_records(queryset)
        return df_track

    elif model_name == 'TrackFeatures':
        queryset = TrackFeatures.objects.select_related('track').all().values(
            'track__valence', 'track__tempo', 'track__speechiness', 
            'track__danceability', 'track__liveness', 'velocity', 
            'median_popularity', 'mean_popularity', 'std_popularity', 
            'trend', 'retrieval_frequency', 'current_popularity'
        )
        df_features = pd.DataFrame.from_records(queryset)
        # Replace None with NaN
        df_features = df_features.replace({None: np.nan})
        # Map categorical features to numerical values
        df_features['retrieval_frequency'] = df_features['retrieval_frequency'].map({
            'high': 2, 'medium': 1, 'low': 0
        })
        
        # Trend mapping
        df_features['trend'] = df_features['trend'].map({
            'up': 1, 'down': -1, 'stable': 0
        })
        
        return df_features