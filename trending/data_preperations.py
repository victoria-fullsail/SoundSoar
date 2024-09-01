import pandas as pd
from .models import TrackFeatures

def fetch_features():
    """
    Fetches track features from the database and returns as a DataFrame.
    """
    features_queryset = TrackFeatures.objects.all()
    data = pd.DataFrame(list(features_queryset.values(
        'danceability', 'energy', 'tempo', 'current_streams',
        'streams_last_24h', 'streams_last_7d', 'streams_last_30d',
        'current_popularity', 'velocity', 'trend'
    )))
    return data

def preprocess_data(data):
    """
    Preprocesses the data: handles categorical data, missing values.
    """
    # Handle categorical data
    data = pd.get_dummies(data, columns=['trend'])

    # Handle missing values
    data.fillna(0, inplace=True)
    
    return data

def prepare_data():
    """
    Fetches and preprocesses data for training.
    """
    data = fetch_features()
    data = preprocess_data(data)
    return data

def split_data(data):
    """
    Splits the data into features and targets.
    """
    X = data.drop(columns=['trend_up', 'trend_down', 'trend_stable'])  # Drop target columns
    y = data[['trend_up', 'trend_down', 'trend_stable']]  # Targets
    
    return X, y

def preprocess_new_data(new_data):
    """
    Preprocesses new data similarly to the training data.
    """
    new_data_df = pd.DataFrame(new_data)
    new_data_df = preprocess_data(new_data_df)
    return new_data_df
