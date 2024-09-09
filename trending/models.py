from django.db import models
from django.utils import timezone
import math


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
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.name
    
class PlaylistTrack(models.Model):
    playlist = models.ForeignKey(Playlist, on_delete=models.CASCADE, related_name='playlist_tracks')
    track = models.ForeignKey('Track', on_delete=models.CASCADE, related_name='playlist_tracks')
    added_at = models.DateTimeField(default=timezone.now)  # When the track was added to the playlist

    def __str__(self):
        return f"{self.track.name} in {self.playlist.name}"

class Track(models.Model):
    spotify_id = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=255)
    album = models.CharField(max_length=255)
    artist = models.CharField(max_length=255)
    popularity = models.IntegerField()
    genre = models.CharField(max_length=255, blank=True, null=True)
    danceability = models.FloatField(blank=True, null=True)
    energy = models.FloatField(blank=True, null=True)
    tempo = models.FloatField(blank=True, null=True)
    valence = models.FloatField(blank=True, null=True)
    speechiness = models.FloatField(blank=True, null=True)
    acousticness = models.FloatField(blank=True, null=True)
    instrumentalness = models.FloatField(blank=True, null=True)
    liveness = models.FloatField(blank=True, null=True)
    added_to_playlists_count = models.IntegerField(default=0)
    updated_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.name} by {self.artist}"

class PopularityHistory(models.Model):
    track = models.ForeignKey(Track, on_delete=models.CASCADE, related_name='popularity_history')
    popularity = models.IntegerField()
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Popularity of {self.track.name} at {self.timestamp}"

class TrackFeatures(models.Model):
    track = models.ForeignKey(Track, on_delete=models.CASCADE, related_name='features')
    danceability = models.FloatField(blank=True, null=True)
    energy = models.FloatField(blank=True, null=True)
    tempo = models.FloatField(blank=True, null=True)
    current_popularity = models.IntegerField(blank=True, null=True)
    popularity_last_3h = models.IntegerField(blank=True, null=True)
    popularity_last_6h = models.IntegerField(blank=True, null=True)
    popularity_last_12h = models.IntegerField(blank=True, null=True)
    popularity_last_24h = models.IntegerField(blank=True, null=True)
    popularity_last_3d = models.IntegerField(blank=True, null=True)
    popularity_last_5d = models.IntegerField(blank=True, null=True)
    velocity = models.FloatField(blank=True, null=True)
    retrieval_frequency = models.CharField(max_length=50, blank=True, null=True, choices=[('high', 'High Frequency'), ('medium', 'Medium Frequency'), ('low', 'Low Frequency')], default='low')
    trend = models.CharField(max_length=50, blank=True, null=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])
    updated_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Features for {self.track.name}"

    def calculate_velocity(self):
        """Calculate the velocity of popularity changes over time."""

        def calculate_rate_of_change(current, past):
            """Calculate the rate of change as a percentage."""
            if current is None or past is None or past == 0:
                return 0
            return (current - past) / past

        # Determine rate of change based on retrieval frequency
        if self.retrieval_frequency == 'high':
            rate_of_change = calculate_rate_of_change(self.popularity_last_3h, self.popularity_last_6h)
        elif self.retrieval_frequency == 'medium':
            rate_of_change = calculate_rate_of_change(self.popularity_last_6h, self.popularity_last_12h)
        elif self.retrieval_frequency == 'low':
            rate_of_change = calculate_rate_of_change(self.popularity_last_24h, self.popularity_last_3d)
        else:
            rate_of_change = 0
        
        self.velocity = rate_of_change
        self.save()


    def calculate_frequency(self):
        """Calculate and update the retrieval frequency based on popularity changes and velocity."""
        # Ensure velocity is up-to-date
        self.calculate_velocity()
        
        # Determine retrieval frequency based on velocity
        if self.velocity > 0.1:  # Adjust threshold as needed
            self.retrieval_frequency = 'high'
        elif 0.01 < self.velocity <= 0.1:  # Adjust threshold as needed
            self.retrieval_frequency = 'medium'
        else:
            self.retrieval_frequency = 'low'
        
        self.save()



    def calculate_trend(self):
        """Calculate and update the trend based on a range of historical data."""
        trends = []
        
        # Compare current popularity with historical data
        if self.popularity_last_6h is not None:
            trends.append('up' if self.current_popularity > self.popularity_last_6h else 'down')
        if self.popularity_last_12h is not None:
            trends.append('up' if self.current_popularity > self.popularity_last_12h else 'down')
        if self.popularity_last_24h is not None:
            trends.append('up' if self.current_popularity > self.popularity_last_24h else 'down')

        # Determine trend based on the most frequent direction
        if trends.count('up') > trends.count('down'):
            self.trend = 'up'
        elif trends.count('down') > trends.count('up'):
            self.trend = 'down'
        else:
            self.trend = 'stable'

        self.save()



    def update_features(self):
        """
        Updates the TrackFeatures instance based on historical popularity data and calculates velocity, trend, and retrieval frequency.
        """
        # Define time periods and their respective cutoff times
        time_periods = {
            '3h': timezone.now() - timezone.timedelta(hours=3),
            '6h': timezone.now() - timezone.timedelta(hours=6),
            '12h': timezone.now() - timezone.timedelta(hours=12),
            '24h': timezone.now() - timezone.timedelta(days=1),
            '3d': timezone.now() - timezone.timedelta(days=3),
            '5d': timezone.now() - timezone.timedelta(days=5),
        }

        # Fetch the most recent popularity entries for each period in one query
        popularity_entries = PopularityHistory.objects.filter(
            track=self.track,
            timestamp__gte=min(time_periods.values())
        ).order_by('-timestamp').values('timestamp', 'popularity')

        # Initialize popularity fields with default values
        popularity_last = {key: {'timestamp': None, 'popularity': None} for key in time_periods.keys()}

        # Populate popularity fields based on fetched data
        for entry in popularity_entries:
            timestamp = entry['timestamp']
            popularity = entry['popularity']
            for period, start_time in time_periods.items():
                if timestamp >= start_time:
                    # Update the field if this entry is more recent
                    if popularity_last[period]['timestamp'] is None or timestamp > popularity_last[period]['timestamp']:
                        popularity_last[period] = {'timestamp': timestamp, 'popularity': popularity}

        # Set fields with fetched data
        self.popularity_last_3h = popularity_last.get('3h')['popularity']
        self.popularity_last_6h = popularity_last.get('6h')['popularity']
        self.popularity_last_12h = popularity_last.get('12h')['popularity']
        self.popularity_last_24h = popularity_last.get('24h')['popularity']
        self.popularity_last_3d = popularity_last.get('3d')['popularity']
        self.popularity_last_5d = popularity_last.get('5d')['popularity']

        # Calculate velocity
        self.calculate_velocity()

        # Determine trend
        self.calculate_trend()

        # Determine retrieval frequency
        self.calculate_frequency()

        # Update the updated_at timestamp and save the instance
        self.updated_at = timezone.now()
        self.save()


    def get_fields_info(self):
        """Returns information about the fields in the TrackFeatures model."""

        regular_fields = [
            'track', 
            'danceability', 
            'energy', 
            'tempo', 
            'current_popularity', 
            'retrieval_frequency',
            'popularity_last_3h', 
            'popularity_last_6h', 
            'popularity_last_12h', 
            'popularity_last_24h', 
            'popularity_last_3d', 
            'popularity_last_5d', 
            'updated_at'
        ]
        
        calculated_fields = [
            'velocity'
        ]
        
        target_fields = [
            'trend'
        ]

        return {
            'regular_fields': regular_fields,
            'calculated_fields': calculated_fields,
            'target_fields': target_fields
        }

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
