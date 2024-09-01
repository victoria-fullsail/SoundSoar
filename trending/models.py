from django.db import models
from django.utils import timezone


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
    streams = models.IntegerField(blank=True, null=True)
    added_to_playlists_count = models.IntegerField(default=0)
    updated_at = models.DateTimeField(default=timezone.now)  # Tracks the last time the data was updated
    stream_change = models.IntegerField(default=0)  # Tracks the change in streams between syncs

    def __str__(self):
        return f"{self.name} by {self.artist}"

    def update_streams(self, new_streams):
        """
        Update the streams and calculate the change in streams.
        This method should be called during your syncing process.
        """
        if self.streams is not None:
            self.stream_change = new_streams - self.streams
        else:
            self.stream_change = new_streams
        self.streams = new_streams
        self.updated_at = timezone.now()
        self.save()

class StreamHistory(models.Model):
    track = models.ForeignKey(Track, on_delete=models.CASCADE, related_name='stream_history')
    timestamp = models.DateTimeField(default=timezone.now)
    streams = models.IntegerField()  # Number of streams at the given timestamp

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.track.name} - {self.timestamp}"

class TrackFeatures(models.Model):
    track = models.ForeignKey(Track, on_delete=models.CASCADE, related_name='features')
    danceability = models.FloatField(blank=True, null=True)
    energy = models.FloatField(blank=True, null=True)
    tempo = models.FloatField(blank=True, null=True)
    current_streams = models.IntegerField()  # Current number of streams
    streams_last_24h = models.IntegerField()  # Streams in the last 24 hours
    streams_last_7d = models.IntegerField()  # Streams in the last 7 days
    streams_last_30d = models.IntegerField()  # Streams in the last 30 days
    current_popularity = models.IntegerField()  # Current popularity score
    velocity = models.FloatField()  # Rate of change in streams
    trend = models.CharField(max_length=50, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')])  # Trend status
    updated_at = models.DateTimeField(default=timezone.now)  # Last update timestamp

    def __str__(self):
        return f"Features for {self.track.name}"

    def update_features(self):
        """Updates the TrackFeatures instance based on the track and stream history."""
        # Update current streams
        self.current_streams = self.track.streams

        # Calculate streams for different periods
        now = timezone.now()
        last_24h = now - timezone.timedelta(days=1)
        last_7d = now - timezone.timedelta(days=7)
        last_30d = now - timezone.timedelta(days=30)

        self.streams_last_24h = self.track.stream_history.filter(timestamp__gte=last_24h).aggregate(models.Sum('streams'))['streams__sum'] or 0
        self.streams_last_7d = self.track.stream_history.filter(timestamp__gte=last_7d).aggregate(models.Sum('streams'))['streams__sum'] or 0
        self.streams_last_30d = self.track.stream_history.filter(timestamp__gte=last_30d).aggregate(models.Sum('streams'))['streams__sum'] or 0

        # Update other features
        self.current_popularity = self.track.popularity
        self.danceability = self.track.danceability
        self.energy = self.track.energy
        self.tempo = self.track.tempo

        # Calculate velocity (rate of change in streams)
        recent_history = self.track.stream_history.filter(timestamp__gte=last_30d).order_by('timestamp')
        if recent_history.count() > 1:
            first_entry = recent_history.first()
            last_entry = recent_history.last()
            self.velocity = (last_entry.streams - first_entry.streams) / (last_entry.timestamp - first_entry.timestamp).total_seconds()
        else:
            self.velocity = 0

        # Determine trend
        if self.current_streams > self.streams_last_7d:
            self.trend = 'up'
        elif self.current_streams < self.streams_last_7d:
            self.trend = 'down'
        else:
            self.trend = 'stable'

        # Save the updated features
        self.save()

    def get_fields_info(self):
        """Returns information about the fields in the TrackFeatures model."""
        regular_fields = ['track', 'updated_at']
        calculated_fields = ['current_streams', 'streams_last_24h', 'streams_last_7d', 'streams_last_30d', 'velocity']
        target_fields = ['trend']

        return {
            'regular_fields': regular_fields,
            'calculated_fields': calculated_fields,
            'target_fields': target_fields
        }

class ModelVersion(models.Model):
    version_number = models.CharField(max_length=50, unique=True)  # E.g., 'v1.0', 'v1.1', 'v2.0'
    description = models.TextField(blank=True, null=True)  # Description of changes or updates
    created_at = models.DateTimeField(default=timezone.now)  # Timestamp when the version was created
    is_active = models.BooleanField(default=False)  # Whether this version is currently in use

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
