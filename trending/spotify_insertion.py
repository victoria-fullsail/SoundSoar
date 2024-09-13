from trending.models import Track, TrackFeatures, PopularityHistory
from django.utils import timezone
import logging

logger = logging.getLogger('spotify')


def insert_or_update_track(track_data):
    """Insert or update a track record in the database."""
    logger.info(f'Inserting or updating track: {track_data["spotify_id"]}')
    track_id = track_data['spotify_id']
    
    track_obj, created = Track.objects.update_or_create(
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
    
    return track_obj


def update_playlist_tracks(playlist, track_data):
    """
    Updates the playlist's tracks with a new list of tracks,
    clearing out any previous tracks.

    Args:
        playlist (Playlist): The playlist to update.
        track_data (list of dict): The list of detailed track data from Spotify API.
    """
    track_ids = []

    for track in track_data:
        spotify_id = track.get('spotify_id')

        if not spotify_id:
            continue  # Skip if the track is invalid or local (no Spotify ID)

        # Update or create the track
        track_obj = insert_or_update_track(track)

        # Insert or update track features
        insert_or_update_track_features(track_obj, track.get('audio_features', {}))

        # Insert popularity history
        insert_popularity_history(track_obj)

        track_ids.append(track_obj.id)

    # Replace the playlist's tracks with the new track list
    playlist.tracks.set(track_ids)
    playlist.save()


def insert_popularity_history(track):
    """Insert a new record into PopularityHistory."""
    logger.debug(f'Inserting popularity history for track: {track.spotify_id}')
    PopularityHistory.objects.create(
        track=track,
        popularity=track.popularity,
        timestamp=timezone.now()
    )


def insert_or_update_track_features(track, audio_features):
    """Insert or update the TrackFeatures record for the given track."""
    logger.info(f'Inserting or updating track features for track: {track.spotify_id}')

    track_features, created = TrackFeatures.objects.update_or_create(
        track=track,
        defaults={
            'valence': audio_features.get('valence', None),
            'energy': audio_features.get('energy', None),
            'tempo': audio_features.get('tempo', None),
            'danceability': audio_features.get('danceability', None),
            'speechiness': audio_features.get('speechiness', None),
            'current_popularity': track.popularity,
            'trend': 'stable',
        }
    )

    track_features.update_features()

    if created:
        logger.info(f'Created new track features for: {track.spotify_id}')
    else:
        logger.info(f'Updated existing track features for: {track.spotify_id}')

    return created
