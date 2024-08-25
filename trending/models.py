from django.db import models

class SocialMediaPlatform(models.Model):
    name = models.CharField(max_length=100, unique=True)
    api_endpoint = models.URLField(max_length=200)
    api_key = models.CharField(max_length=255, blank=True, null=True)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Social Media Platform'
        verbose_name_plural = 'Social Media Platforms'

class Sound(models.Model):
    """
    Model to store top sounds from various social media platforms.
    """
    # Define the basic emotions as choices
    JOY = 'joy'
    SADNESS = 'sadness'
    ANGER = 'anger'
    FEAR = 'fear'
    SURPRISE = 'surprise'
    DISGUST = 'disgust'

    BASIC_EMOTIONS = [
        (JOY, 'Joy'),
        (SADNESS, 'Sadness'),
        (ANGER, 'Anger'),
        (FEAR, 'Fear'),
        (SURPRISE, 'Surprise'),
        (DISGUST, 'Disgust'),
    ]

    platform = models.ForeignKey(SocialMediaPlatform, on_delete=models.CASCADE, related_name='top_sounds')
    sentiment = models.CharField(
        max_length=10,
        choices=BASIC_EMOTIONS,
        default=JOY,  # Default to 'Joy' or any other emotion you prefer
    )
    sound_name = models.CharField(max_length=255)
    sound_url = models.URLField(max_length=255, blank=True, null=True)
    usage_count = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.sound_name} ({self.platform.name})"