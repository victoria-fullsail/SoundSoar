from django.db import models
from django.utils import timezone
from datetime import timedelta
import numpy as np


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
    
class Playlist(models.Model):
    chart = models.ForeignKey(Chart, on_delete=models.CASCADE, related_name='playlists')
    playlist_id = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    tracks = models.ManyToManyField('Track', related_name='playlists')
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.name

class Track(models.Model):
    spotify_id = models.CharField(max_length=100, unique=True)
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

    valence = models.FloatField(blank=True, null=True)
    energy = models.FloatField(blank=True, null=True)
    tempo = models.FloatField(blank=True, null=True)
    danceability = models.FloatField(blank=True, null=True)
    speechiness = models.FloatField(blank=True, null=True)

    current_popularity = models.IntegerField(blank=True, null=True)

    velocity = models.FloatField(blank=True, null=True)
    median_popularity = models.FloatField(blank=True, null=True)
    mean_popularity = models.FloatField(blank=True, null=True)
    std_popularity = models.FloatField(blank=True, null=True)
    trend = models.CharField(max_length=50, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])

    retrieval_frequency = models.CharField(max_length=50, blank=True, null=True, choices=[('high', 'High Frequency'), ('medium', 'Medium Frequency'), ('low', 'Low Frequency')], default='low')
    updated_at = models.DateTimeField(default=timezone.now)

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

    def get_trend_model_features_and_target():
        pass

    def get_popularity_model_features_and_target():
        pass


class ModelVersion(models.Model):
    version_number = models.CharField(max_length=25, unique=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return f"Version {self.version_number} - {'Active' if self.is_active else 'Inactive'}"

    def activate(self):
        """Activate this version and deactivate all others."""
        ModelVersion.objects.update(is_active=False)  # Deactivate all other versions
        self.is_active = True
        self.save()

    def deactivate(self):
        """Deactivate this version."""
        self.is_active = False
        self.save()

class ModelPerformance(models.Model):
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='performances')
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    evaluation_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Performance for {self.model_version.version_number} on {self.evaluation_date}"
