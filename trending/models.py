from django.db import models
from django.utils import timezone
from datetime import timedelta
import numpy as np
import pandas as pd
from .use_trend_model import load_active_model 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from django.db import transaction
from collections import Counter

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
    lda_prediction = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])
    extra_prediction = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])
    knn_prediction = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])

    predicted_trend = models.CharField(max_length=10, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])

    def __str__(self):
        return f"Features for {self.track.name}"

    def calculate_velocity(self):
        """
        Calculate the velocity of popularity based on historical data.

        Velocity is defined as the rate of change of popularity over time.
        If there are less than two historical popularity values, velocity is set to 0.

        Attributes:
            velocity (float): The calculated velocity of popularity change.
        """
        historical_popularity = self.get_historical_popularity()
        if len(historical_popularity) < 2:
            self.velocity = 0
            return
            
        recent = np.array(historical_popularity[-2:])
        rate_of_change = (recent[1] - recent[0]) / recent[0] if recent[0] != 0 else 0
        self.velocity = rate_of_change

    def calculate_mean(self):
        historical_popularity = self.get_historical_popularity()
        self.mean_popularity = np.mean(historical_popularity) if historical_popularity else None

    def calculate_median(self):
        historical_popularity = self.get_historical_popularity()
        self.median_popularity = np.median(historical_popularity) if historical_popularity else None

    def calculate_std(self):
        historical_popularity = self.get_historical_popularity()
        self.std_popularity = np.std(historical_popularity) if historical_popularity else None

    def calculate_trend(self):
        """
        Determines the trend of current popularity compared to historical data.
        A trend can be 'up', 'down', or 'stable' based on:
        - Current popularity vs. mean popularity
        - Rate of change (velocity) of historical popularity
        """

        # Get historical popularity data
        historical_popularity = self.get_historical_popularity()
        
        # If there are not enough data points to determine a trend, default to 'stable'
        if len(historical_popularity) < 3:
            self.trend = 'stable'
            return

        # Use the previously calculated mean popularity
        mean_popularity = self.mean_popularity
        
        # Compare current popularity with mean popularity
        if self.current_popularity > mean_popularity:
            # Determine trend direction based on velocity
            if self.velocity > 0:
                self.trend = 'up'  # Upward trend
            elif self.velocity == 0:
                self.trend = 'stable'  # No significant change
            else:
                self.trend = 'down'  # Current popularity is decreasing

        elif self.current_popularity < mean_popularity:
            # Determine trend direction based on velocity
            if self.velocity < 0:
                self.trend = 'down'  # Downward trend
            elif self.velocity == 0:
                self.trend = 'stable'  # No significant change
            else:
                self.trend = 'up'  # Current popularity is improving

        else:
            self.trend = 'stable'  # Current popularity equals mean

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
            'median_popularity': self.median_popularity if self.median_popularity is not None else self.current_popularity,
            'mean_popularity': self.mean_popularity if self.mean_popularity is not None else self.current_popularity,
            'std_popularity': self.std_popularity if self.std_popularity is not None else 0,
            'retrieval_frequency': retrieval_frequency
        }

        features_df = pd.DataFrame([features])
        print('features_df: ', features_df)

        # Load the active RandomForest model to get the imputer (or any other active model with saved imputer)
        try:
            rf_model = TrendModel.objects.get(model_type='RandomForest', is_active=True)
            _, feature_names, imputer = load_active_model(rf_model)
        except TrendModel.DoesNotExist:
            print("Error: No active RandomForest model found.")
            return None
        except Exception as e:
            print(f"An error occurred while fetching the RandomForest model: {e}")
            return None

        # Apply the imputer loaded with the model
        features_imputed = pd.DataFrame(imputer.transform(features_df), columns=features_df.columns)
        print('features_imputed: ', features_imputed)

        # Load all active models
        active_models = {
            'rf': TrendModel.objects.get(model_type='RandomForest', is_active=True),
            'hgb': TrendModel.objects.get(model_type='HistGradientBoosting', is_active=True),
            'lr': TrendModel.objects.get(model_type='LogisticRegression', is_active=True),
            'svm': TrendModel.objects.get(model_type='SVM', is_active=True),
            'lda': TrendModel.objects.get(model_type='LDA', is_active=True),
            'et': TrendModel.objects.get(model_type='ExtraTrees', is_active=True),
            'knn': TrendModel.objects.get(model_type='KNN', is_active=True)
        }

        predictions = {}

        # Iterate over the models and make predictions
        for model_name, model_instance in active_models.items():
            try:
                model, feature_names, _ = load_active_model(model_instance)

                # If the model is Logistic Regression or SVM, scale the features
                if model_name in ['lr', 'svm']:
                    scaler = StandardScaler()
                    features_scaled = pd.DataFrame(scaler.fit_transform(features_imputed), columns=features_imputed.columns)
                    features_for_model = features_scaled[feature_names]
                else:
                    # No scaling needed for other models
                    features_for_model = features_imputed[feature_names]

                # Make the prediction
                predictions[model_name] = self.map_prediction(model.predict(features_for_model.values)[0])

            except Exception as e:
                print(f"An error occurred while predicting with {model_name}: {e}")
                predictions[model_name] = None

        # Update the prediction fields
        self.rf_prediction = predictions['rf']
        self.hgb_prediction = predictions['hgb']
        self.lr_prediction = predictions['lr']
        self.svm_prediction = predictions['svm']
        self.lda_prediction = predictions['lda']
        self.extra_prediction = predictions['et']
        self.knn_prediction = predictions['knn']


        # Determine the most common prediction
        most_common_prediction = max(set(predictions.values()), key=list(predictions.values()).count)
        self.predicted_trend = most_common_prediction

        self.save()

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

class TrendModel(models.Model):
    version_number = models.CharField(max_length=25, unique=True, blank=True)
    model_order = models.IntegerField(blank=True, null=True)
    phase_number = models.IntegerField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=False)
    is_best = models.BooleanField(default=False)
    model_type = models.CharField(max_length=25, choices=[
        ('RandomForest', 'RandomForest'), 
        ('HistGradientBoosting', 'HistGradientBoosting'),
        ('LogisticRegression', 'LogisticRegression'),
        ('SVM', 'SVM'),
        ('LDA', 'LDA'),
        ('ExtraTrees', 'ExtraTrees'),
        ('KNN', 'KNN'),
    ])
    accuracy = models.FloatField(blank=True, null=True)
    precision = models.FloatField(blank=True, null=True)
    recall = models.FloatField(blank=True, null=True)
    f1_score = models.FloatField(blank=True, null=True)
    roc_auc = models.FloatField(null=True, blank=True)
    best_parameters = models.TextField(blank=True, null=True)
    confusion_matrix = models.TextField(blank=True, null=True)
    csv_data = models.FileField(upload_to='trending/trend_model/csv_data/', null=True, blank=True)
    model_file = models.FileField(upload_to='trending/trend_model/models/', null=True, blank=True)
    readme_file = models.FileField(upload_to='trending/trend_model/readme/', null=True, blank=True)
    evaluation_date = models.DateTimeField(default=timezone.now)
    popularity_timeframe = models.CharField(max_length=40, blank=True, null=True)

    @staticmethod
    def get_parameters_count_list_for_type(model_type='RandomForest'):
        models = TrendModel.objects.filter(model_type=model_type).exclude(best_parameters__isnull=True).exclude(best_parameters__exact='')

        if not models.exists():
            return []  # Return an empty list if no models with best parameters exist

        # Extract all best_parameters into a list
        parameter_list = [model.best_parameters for model in models]

        # Use Counter to count the occurrences
        parameter_counter = Counter(parameter_list)

        # Convert the counter to a list of tuples (parameter, count) sorted by count
        parameter_count_list = parameter_counter.most_common()

        return parameter_count_list
  
    @staticmethod
    def get_average_score_for_type(model_type='RandomForest', metric='accuracy'):
        models = TrendModel.objects.filter(model_type=model_type)

        if not models.exists():
            return 0.0  # Return 0 if there are no models of this type

        total_score = 0.0
        count = 0

        for model in models:
            if metric == 'accuracy' and model.accuracy is not None:
                total_score += model.accuracy
                count += 1
            elif metric == 'precision' and model.precision is not None:
                total_score += model.precision
                count += 1
            elif metric == 'recall' and model.recall is not None:
                total_score += model.recall
                count += 1
            elif metric == 'f1_score' and model.f1_score is not None:
                total_score += model.f1_score
                count += 1
            # Add other metrics here if needed

        if count == 0:
            return 0.0  # Return 0 if no valid scores were found

        average_score = total_score / count
        return average_score
             
    def _get_historical_dates(self):
        """Calculate the start and end dates for the historical data (last 30 days)."""
        end_date = self.created_at
        start_date = end_date - timedelta(days=30)
        return start_date, end_date

    def calculate_popularity_timeframe(self):
        """Calculate the popularity timeframe string."""
        start_date, end_date = self._get_historical_dates()
        return f"{start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')}"

    def generate_filename(self, file_type='csv'):
        """Generate a filename for the trend model based on the version number and date range."""
        start_date, end_date = self._get_historical_dates()
        return f"trend_model_{self.version_number}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.{file_type}"

    def calculate_phase_order_string(self):
        if self.phase_number and self.model_order:
            return f"{self.phase_number}.{self.model_order}"
        else:
            return 'Not Available'
        
    @staticmethod
    def calculate_new_phase():
        # Get the last phase number, if it exists
        last_phase = TrendModel.objects.filter(phase_number__isnull=False).order_by('-phase_number').first()
        
        if last_phase:
            # Increment the last phase number by 1 for the new instance
            return last_phase.phase_number + 1
        
        return 1

    def save(self, *args, **kwargs):
        # Version Number
        if not self.version_number:
            with transaction.atomic():
                last_model = TrendModel.objects.order_by('created_at').last()
                if last_model and last_model.version_number:
                    last_version_num = int(last_model.version_number.split('.')[0][1:])
                    proposed_version = f'v{last_version_num + 1}.0'
                    while TrendModel.objects.filter(version_number=proposed_version).exists():
                        last_version_num += 1
                        proposed_version = f'v{last_version_num}.0'
                    self.version_number = proposed_version
                else:
                    self.version_number = 'v1.0'

        # Set Popularity History Timeframe
        self.popularity_timeframe = self.calculate_popularity_timeframe()

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

class FeatureImportance(models.Model):
    trend_model = models.ForeignKey(TrendModel, on_delete=models.CASCADE, related_name='feature_importances')
    feature_name = models.CharField(max_length=50)
    importance = models.FloatField()

    def __str__(self):
            return f"Feature Importance of {self.feature_name} for {self.trend_model}."

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
