from trending.models import Track, TrackFeatures, PopularityHistory
from django.utils import timezone
import logging

# Get the Spotify logger
logger = logging.getLogger('spotify')


def insert_or_update_track(track_data):
    """Insert a new track or update an existing one based on Spotify data."""
    logger.debug(f'Inserting or updating track: {track_data["spotify_id"]}')
    track_id = track_data['spotify_id']
    
    # Create or update the Track record
    track, created = Track.objects.update_or_create(
        spotify_id=track_id,
        defaults={
            'name': track_data['name'],
            'album': track_data['album'],
            'artist': track_data['artists'],
            'popularity': track_data['popularity'],
            'danceability': track_data['audio_features'].get('danceability', None),
            'energy': track_data['audio_features'].get('energy', None),
            'tempo': track_data['audio_features'].get('tempo', None),
            'valence': track_data['audio_features'].get('valence', None),
            'speechiness': track_data['audio_features'].get('speechiness', None),
            'acousticness': track_data['audio_features'].get('acousticness', None),
            'instrumentalness': track_data['audio_features'].get('instrumentalness', None),
            'liveness': track_data['audio_features'].get('liveness', None),
            'updated_at': timezone.now()
        }
    )
    
    if created:
        logger.info(f'Created new track: {track_id}')
    else:
        logger.info(f'Updated existing track: {track_id}')
    
    return track


def insert_or_update_popularity_history(track):
    """Insert a new record into PopularityHistory."""
    logger.debug(f'Inserting popularity history for track: {track.spotify_id}')
    PopularityHistory.objects.create(
        track=track,
        popularity=track.popularity,
        timestamp=timezone.now()
    )


def insert_or_update_track_features(track, audio_features):
    """Insert or update the TrackFeatures record for the given track."""
    logger.debug(f'Inserting or updating track features for track: {track.spotify_id}')

    track_features, created = TrackFeatures.objects.update_or_create(
        track=track,
        defaults={
            'danceability': audio_features.get('danceability', None),
            'energy': audio_features.get('energy', None),
            'tempo': audio_features.get('tempo', None),
            'current_popularity': track.popularity,
            'trend': 'stable',
            'retrieval_frequency': 'low'
        }
    )

    # Update features with calculations
    track_features.update_features()

    if created:
        logger.info(f'Created new track features for: {track.spotify_id}')
    else:
        logger.info(f'Updated existing track features for: {track.spotify_id}')

    return created


def process_and_insert_data(detailed_tracks):
    """Process and insert/update all tracks and their details into the database."""
    logger.info('Processing and inserting track data.')

    for track_data in detailed_tracks:
        try:
            # Extract audio features
            audio_features = track_data['audio_features']
            
            # Insert or update the Track record
            track = insert_or_update_track(track_data)

            # Insert or update the PopularityHistory record
            insert_or_update_popularity_history(track)
            
            # Insert or update the TrackFeatures record
            insert_or_update_track_features(track, audio_features)
        
        except Exception as e:
            logger.error(f'Error processing track data for {track_data["spotify_id"]}: {e}', exc_info=True)
