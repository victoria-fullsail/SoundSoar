from django.db import models
from django.utils import timezone
from datetime import timedelta
import numpy as np
import pandas as pd
from .use_trend_model import load_active_model 


class Chart(models.Model):
    CHART_TYPE_CHOICES = [
        ('custom', 'Custom'),
        ('spotify_playlist', 'Spotify Playlist'),
    ]
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    chart_type = models.CharField(max_length=20, choices=CHART_TYPE_CHOICES, default='spotify_playlist')

    def __str__(self):
        return f"{self.name} ({self.get_chart_type_display()})"
    
class CustomPlaylist(models.Model):
    chart = models.ForeignKey(Chart, on_delete=models.CASCADE, related_name='custom_playlists')
    name = models.CharField(max_length=255)
    tracks = models.ManyToManyField('Track', related_name='custom_playlists', blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.name
    
    def update_tracks(self, required_count=25, threshold_popularity=65):

        # Clear existing tracks
        self.tracks.clear()

        # Query Top tracks by popularity, up, stable
        high_pop_up_stable_features = TrackFeatures.objects.filter(
            current_popularity__gte=threshold_popularity,
            predicted_trend__in=['up']
        ).order_by('-current_popularity')[:required_count]

        high_pop_up_stable_tracks = [feature.track for feature in high_pop_up_stable_features]
        self.tracks.set(high_pop_up_stable_tracks)

        self.save()

class Playlist(models.Model):
    chart = models.ForeignKey(Chart, on_delete=models.CASCADE, related_name='playlists')
    playlist_id = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=255)
    tracks = models.ManyToManyField('Track', related_name='playlists')
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.name

class Track(models.Model):
    spotify_id = models.CharField(max_length=100, unique=True)
    spotify_url = models.URLField(blank=True, null=True)
    name = models.CharField(max_length=255)
    album = models.CharField(max_length=255)
    artist = models.CharField(max_length=255)
    popularity = models.IntegerField()
    danceability = models.FloatField(blank=True, null=True)
    energy = models.FloatField(blank=True, null=True)
    tempo = models.FloatField(blank=True, null=True)
    valence = models.FloatField(blank=True, null=True)
    speechiness = models.FloatField(blank=True, null=True)
    acousticness = models.FloatField(blank=True, null=True)
    instrumentalness = models.FloatField(blank=True, null=True)
    liveness = models.FloatField(blank=True, null=True)
    updated_at = models.DateTimeField(default=timezone.now)
 
    def __str__(self):
        return f"{self.name} by {self.artist}"

class PopularityHistory(models.Model):
    track = models.ForeignKey(Track, on_delete=models.CASCADE, related_name='popularity_history')
    popularity = models.IntegerField()
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-timestamp']  # Order by most recent first

    def __str__(self):
        return f"Popularity of {self.track.name} at {self.timestamp}"

class TrackFeatures(models.Model):
    track = models.ForeignKey(Track, on_delete=models.CASCADE, related_name='features')
    current_popularity = models.IntegerField(blank=True, null=True)
    velocity = models.FloatField(blank=True, null=True)
    median_popularity = models.FloatField(blank=True, null=True)
    mean_popularity = models.FloatField(blank=True, null=True)
    std_popularity = models.FloatField(blank=True, null=True)
    trend = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])
    retrieval_frequency = models.CharField(max_length=50, blank=True, null=True, choices=[('high', 'High Frequency'), ('medium', 'Medium Frequency'), ('low', 'Low Frequency')], default='low')
    updated_at = models.DateTimeField(default=timezone.now)

    rf_prediction = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])
    hgb_prediction = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])
    lr_prediction = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])
    svm_prediction = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])

    predicted_trend = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])


    def __str__(self):
        return f"Features for {self.track.name}"

    def calculate_velocity(self):
        # Assuming you have a method to get historical popularity data
        historical_popularity = self.get_historical_popularity()
        if len(historical_popularity) < 2:
            self.velocity = 0
            return
        
        recent = np.array(historical_popularity[-2:])
        rate_of_change = (recent[1] - recent[0]) / recent[0] if recent[0] != 0 else 0
        self.velocity = rate_of_change

    def calculate_mean(self):
        # Assuming you have a method to get historical popularity data
        historical_popularity = self.get_historical_popularity()
        self.mean_popularity = np.mean(historical_popularity) if historical_popularity else None

    def calculate_median(self):
        # Assuming you have a method to get historical popularity data
        historical_popularity = self.get_historical_popularity()
        self.median_popularity = np.median(historical_popularity) if historical_popularity else None

    def calculate_std(self):
        # Assuming you have a method to get historical popularity data
        historical_popularity = self.get_historical_popularity()
        self.std_popularity = np.std(historical_popularity) if historical_popularity else None

    def calculate_trend(self):
        # Use median, mean, or any other relevant statistics for trend determination
        if self.mean_popularity and self.current_popularity:
            if self.current_popularity > self.mean_popularity:
                self.trend = 'up'
            elif self.current_popularity < self.mean_popularity:
                self.trend = 'down'
            else:
                self.trend = 'stable'
        else:
            self.trend = 'stable'

    def calculate_frequency(self):
        # Determine retrieval frequency based on velocity
        if self.velocity > 0.1:
            self.retrieval_frequency = 'high'
        elif 0.01 < self.velocity <= 0.1:
            self.retrieval_frequency = 'medium'
        else:
            self.retrieval_frequency = 'low'

    def update_features(self):
        """ Updates the TrackFeatures instance based on historical popularity data and calculates velocity, trend, and retrieval frequency. """
        self.calculate_velocity()
        self.calculate_trend()
        self.calculate_frequency()
        self.calculate_mean()
        self.calculate_median()
        self.calculate_std()

        # Update the updated_at timestamp and save the instance
        self.updated_at = timezone.now()
        self.save()

    def predict_and_update_trend(self):
        """
        Use the trained models to predict the trend and update the predictions in the model.
        Determine the overall predicted trend based on the most common prediction.
        """
        # Prepare the input features for prediction
        retrieval_frequency_mapping = {'high': 2, 'medium': 1, 'low': 0}
        retrieval_frequency = retrieval_frequency_mapping.get(self.retrieval_frequency, np.nan)

        features = {
            'track__valence': self.track.valence,
            'track__tempo': self.track.tempo,
            'track__speechiness': self.track.speechiness,
            'track__danceability': self.track.danceability,
            'track__liveness': self.track.liveness,
            'velocity': self.velocity,
            'current_popularity': self.current_popularity,
            'median_popularity': self.median_popularity,
            'mean_popularity': self.mean_popularity,
            'std_popularity': self.std_popularity,
            'retrieval_frequency': retrieval_frequency
        }

        features_df = pd.DataFrame([features])

        # Store predictions from each model
        predictions = {
            'rf': self.predict_rf(features_df),
            'hgb': self.predict_hgb(features_df),
            'lr': self.predict_lr(features_df),
            'svm': self.predict_svm(features_df)
        }

        # Update the prediction fields
        self.rf_prediction = predictions['rf']
        self.hgb_prediction = predictions['hgb']
        self.lr_prediction = predictions['lr']
        self.svm_prediction = predictions['svm']

        # Determine the most common prediction
        most_common_prediction = max(set(predictions.values()), key=predictions.values().count)
        self.predicted_trend = most_common_prediction

        self.save()

    def predict_rf(self, features_df):
        rf_model = TrendModel.objects.get(model_type='RandomForest', is_active=True)
        print('Predicting with RF...')
        model, feature_names = load_active_model(rf_model)
        return self.map_prediction(model.predict(features_df[feature_names])[0])

    def predict_hgb(self, features_df):
        hgb_model = TrendModel.objects.get(model_type='HistGradientBoost', is_active=True)
        print('Predicting with HGB...')
        model, feature_names = load_active_model(hgb_model)
        return self.map_prediction(model.predict(features_df[feature_names])[0])

    def predict_lr(self, features_df):
        lr_model = TrendModel.objects.get(model_type='LogisticRegression', is_active=True)
        print('Predicting with LR...')
        model, feature_names = load_active_model(lr_model)
        return self.map_prediction(model.predict(features_df[feature_names])[0])

    def predict_svm(self, features_df):
        svm_model = TrendModel.objects.get(model_type='SVM', is_active=True)
        print('Predicting with SVM...')
        model, feature_names = load_active_model(svm_model)
        return self.map_prediction(model.predict(features_df[feature_names])[0])

    def map_prediction(self, prediction):
        """
        Map numeric predictions to string labels.
        """
        if prediction == 1:
            return 'up'
        elif prediction == -1:
            return 'down'
        elif prediction == 0:
            return 'stable'
        return None

    def get_historical_popularity(self, days=30):
        """
        Retrieves historical popularity data for the last `days` days.
        """
        # Get the current timestamp
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)

        # Query to get historical popularity data for the track
        historical_data = PopularityHistory.objects.filter(
            track=self.track,
            timestamp__range=(start_date, end_date)
        ).values_list('popularity', flat=True).order_by('timestamp')

        return list(historical_data)

    def get_historical_popularity_tuples(self, days=30):
        """
        Retrieves historical popularity data for the last `days` days as a list of tuples (timestamp, popularity).
        """
        # Get the current timestamp
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)

        # Query to get historical popularity data for the track
        historical_data = PopularityHistory.objects.filter(
            track=self.track,
            timestamp__range=(start_date, end_date)
        ).order_by('timestamp')

        # Create a list of tuples (timestamp, popularity)
        return [(entry.timestamp, entry.popularity) for entry in historical_data]

    def get_trend_model_features_and_target():
        pass

    def get_popularity_model_features_and_target():
        pass

class TrendModel(models.Model):
    version_number = models.CharField(max_length=25, unique=True, blank=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=False)
    is_best = models.BooleanField(default=False)
    model_type = models.CharField(max_length=25, choices=[
        ('RandomForest', 'RandomForest'), 
        ('HistGradientBoost', 'HistGradientBoost'),
        ('LogisticRegression', 'LogisticRegression'),
        ('SVM', 'SVM')  
    ])
    accuracy = models.FloatField(blank=True, null=True)
    precision = models.FloatField(blank=True, null=True)
    recall = models.FloatField(blank=True, null=True)
    f1_score = models.FloatField(blank=True, null=True)
    roc_auc = models.FloatField(null=True, blank=True)
    evaluation_date = models.DateTimeField(default=timezone.now)

    def save(self, *args, **kwargs):
        # Automatically generate a unique version number if not set
        if not self.version_number:
            last_model = TrendModel.objects.order_by('created_at').last()
            if last_model and last_model.version_number:
                last_version_num = int(last_model.version_number.split('.')[0][1:])
                self.version_number = f'v{last_version_num + 1}.0'
            else:
                # If this is the first version, start with 'v1.0'
                self.version_number = 'v1.0'
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Version {self.version_number} - {self.model_type} ({'Active' if self.is_active else 'Inactive'})"

    def activate(self):
        """Activate this model and deactivate all others."""
        self.is_active = True
        self.save()

    def deactivate(self):
        """Deactivate this model."""
        self.is_active = False
        self.save()

class PredictionHistory(models.Model):
    trend_model = models.ForeignKey(TrendModel, on_delete=models.CASCADE, related_name='predictions')
    song = models.ForeignKey(Track, on_delete=models.CASCADE, related_name='trend_predictions')
    predicted_trend = models.CharField(max_length=50, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])
    actual_trend = models.CharField(max_length=50, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')], blank=True, null=True)
    predicted_at = models.DateTimeField(default=timezone.now)
    actualized_at = models.DateTimeField(blank=True, null=True)
    
    def is_correct(self):
        """Check if the prediction was correct"""
        return self.predicted_trend == self.actual_trend

    def __str__(self):
        return f"Prediction for {self.song.name}: {self.predicted_trend} (Actual: {self.actual_trend if self.actual_trend else 'Pending'})"
