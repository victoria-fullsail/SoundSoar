import os
import django
import sys

# Set up the project base directory and Django settings module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# Initialize Django
django.setup()


from trending.models import TrackFeatures, Track, PopularityHistory, Playlist
from spotify_api import fetch_playlist_with_details, fetch_track_details
from spotify_insertion import process_and_insert_data
from django.utils import timezone
import logging

# Get the Spotify logger
logger = logging.getLogger('spotify')

def high_freq_track_popularity_history_task():
    """Task to update popularity history for tracks with specific retrieval frequency."""
    # Fetch all tracks that require high frequency updates
    tracks = TrackFeatures.objects.filter(retrieval_frequency='high')

    for track_features in tracks:
        track = track_features.track
        
        try:
            # Fetch the current details from Spotify
            track_details = fetch_track_details(track.spotify_id)
            current_popularity = track_details['track_info']['popularity']

            # Update the popularity in the Track model
            track.popularity = current_popularity
            track.save()

            # Update the popularity history
            PopularityHistory.objects.create(
                track=track,
                popularity=current_popularity,
                timestamp=timezone.now()
            )
            
            # Update the TrackFeatures record with new details
            track_features.update_features()
            
            # Log the update status
            logger.info(f'Updated track: {track.spotify_id}')

        except Exception as e:
            logger.error(f'Error updating track {track.spotify_id}: {e}')
    
    logger.info('High frequency track popularity history task completed.')

def medium_freq_track_popularity_history_task():
    """Task to update popularity history for tracks with specific retrieval frequency."""
    # Fetch all tracks that require medium frequency updates
    tracks = TrackFeatures.objects.filter(retrieval_frequency='medium')

    for track_features in tracks:
        track = track_features.track
        
        try:
            # Fetch the current details from Spotify
            track_details = fetch_track_details(track.spotify_id)
            current_popularity = track_details['track_info']['popularity']

            # Update the popularity in the Track model
            track.popularity = current_popularity
            track.save()

            # Update the popularity history
            PopularityHistory.objects.create(
                track=track,
                popularity=current_popularity,
                timestamp=timezone.now()
            )
            
            # Update the TrackFeatures record with new details
            track_features.update_features()
            
            # Log the update status
            logger.info(f'Updated track: {track.spotify_id}')

        except Exception as e:
            logger.error(f'Error updating track {track.spotify_id}: {e}')
    
    logger.info('Medium frequency track popularity history task completed.')

def low_freq_track_popularity_history_task():
    """Task to update popularity history for tracks with specific retrieval frequency."""
    # Fetch all tracks that require low frequency updates
    tracks = TrackFeatures.objects.filter(retrieval_frequency='low')

    for track_features in tracks:
        track = track_features.track
        
        try:
            # Fetch the current details from Spotify
            track_details = fetch_track_details(track.spotify_id)
            current_popularity = track_details['track_info']['popularity']

            # Update the popularity in the Track model
            track.popularity = current_popularity
            track.save()

            # Update the popularity history
            PopularityHistory.objects.create(
                track=track,
                popularity=current_popularity,
                timestamp=timezone.now()
            )
            
            # Update the TrackFeatures record with new details
            track_features.update_features()
            
            # Log the update status
            logger.info(f'Updated track: {track.spotify_id}')

        except Exception as e:
            logger.error(f'Error updating track {track.spotify_id}: {e}')
    
    logger.info('Low frequency track popularity history task completed.')

def update_or_create_track_data_task():
    """Task to retrieve data from Spotify playlists and insert/update track data in the database."""
    # Fetch all playlists from your database (assuming Playlist contains the playlists)
    playlists = Playlist.objects.all()  # Or filter specific playlists if needed
    #playlist = Playlist.objects.first()
    for playlist in playlists:
        try:
            # Fetch the playlist details from Spotify
            logger.info(f'Fetching details for playlist: {playlist.playlist_id}')
            detailed_tracks = fetch_playlist_with_details(playlist.playlist_id)
            
            # Insert or update track data in the database
            process_and_insert_data(detailed_tracks)
            logger.info(f'Successfully updated playlist: {playlist.playlist_id}')

        except Exception as e:
            logger.error(f'Error processing playlist {playlist.playlist_id}: {e}')

if __name__ == "__main__":

    # Run the function
    print("Starting to update or create track data from Spotify playlists...")
    try:
        update_or_create_track_data_task()
        print("Successfully updated or created track data.")
    except Exception as e:
        print(f"Error during track data update: {e}")
