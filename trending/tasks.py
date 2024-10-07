import os
import sys
import django
import logging
import argparse

# Set up the project base directory and Django settings module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
logger = logging.getLogger('spotify')

# Import models and functions
from trending.models import TrackFeatures, Playlist, CustomPlaylist
from spotify_api import fetch_playlist_with_details, fetch_track_details
from spotify_insertion import update_playlist_tracks, insert_or_update_track, insert_or_update_track_features, insert_popularity_history
from update_trend_model import update_active_models


def sync_playlist_tracks_task():
    """Sync playlist tracks every 24 hours."""

    playlists = Playlist.objects.all()

    for playlist in playlists:

        try:
            logger.info(f'Fetching track ids for playlist id: {playlist.playlist_id}')
            track_data = fetch_playlist_with_details(playlist.playlist_id)

            update_playlist_tracks(playlist, track_data)
            logger.info(f'Successfully updated playlist: {playlist.playlist_id}')

        except Exception as e:
            logger.error(f'Error processing playlist {playlist.playlist_id}: {e}')


def high_freq_sync_track_data_task():
    """High-frequency sync for certain tracks every 6 hours"""
    # Fetch high-frequency track features from the database
    high_freq_tracks = TrackFeatures.objects.filter(retrieval_frequency='high')

    for track_features in high_freq_tracks:
        track = track_features.track

        # Fetch the latest track details
        track_data = fetch_track_details(track.spotify_id)

        # Update or insert track data
        insert_or_update_track(track_data)

        # Update or insert track features
        insert_or_update_track_features(track, track_data['audio_features'])

        # Insert or update popularity history
        insert_popularity_history(track)

    logger.info("High-frequency track data synchronization completed.")


def medium_freq_sync_track_data_task():
    """Medium-frequency sync for certain tracks every 12 hours"""

    # Fetch medium-frequency track features from the database
    medium_freq_tracks = TrackFeatures.objects.filter(retrieval_frequency='medium')

    for track_features in medium_freq_tracks:
        track = track_features.track

        # Fetch the latest track details
        track_data = fetch_track_details(track.spotify_id)

        # Update or insert track data
        insert_or_update_track(track_data)

        # Update or insert track features
        insert_or_update_track_features(track, track_data['audio_features'])

        # Insert or update popularity history
        insert_popularity_history(track)

    logger.info("Medium-frequency track data synchronization completed.")

def update_trend_models():
    # Update model
    update_active_models()

def update_all_track_features_predictions():
    """
    Updates the predicted_trend field for all TrackFeatures in the database.
    """

    # Update predictions
    track_features = TrackFeatures.objects.all()

    for feature in track_features:
        try:
            print(f"Updated predicted trend for {feature.track.name} which is {feature.predicted_trend}")
            feature.predict_and_update_trend()  # Run the prediction and update         
        except Exception as e:
            print(f"Error updating trend for {feature.track.name}: {e}")


def update_custom_playlist():
    try:
        # Get the playlist by its ID (SoundSoar Suggestions ID = 1)
        playlist = CustomPlaylist.objects.get(id=1)
        
        # Update the tracks
        playlist.update_tracks()

        print(f"Playlist '{playlist.name}' updated successfully.")
        
    except CustomPlaylist.DoesNotExist:
        print(f"CustomPlaylist with ID {1} does not exist.")


def main():
    parser = argparse.ArgumentParser(description="Run specific functions.")
    parser.add_argument('function', choices=['sync_playlist', 'high_freq', 'medium_freq', 'update_models', 'update_predictions', 'update_custom_playlist'], help="Function to run")
    args = parser.parse_args()

    if args.function == 'sync_playlist':
        sync_playlist_tracks_task()
    elif args.function == 'high_freq':
        high_freq_sync_track_data_task()
    elif args.function == 'medium_freq':
        medium_freq_sync_track_data_task()
        print('models...')
    elif args.function == 'update_models':
        update_trend_models()
    elif args.function == 'update_predictions':
        print('predictions...')
        update_all_track_features_predictions()
    elif args.function == 'update_custom_playlist':
        update_custom_playlist()

if __name__ == "__main__":
    main()