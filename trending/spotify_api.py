import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging
from decouple import config, Config, RepositoryEnv
import os

logger = logging.getLogger('spotify')


def get_spotify_client():
    """Authenticate with Spotify using client credentials."""

    # Get the directory of the current script and then it's parent
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Construct the path to the .env file relative to the parent directory
    env_path = os.path.join(parent_dir, 'config', '.env')

    # Create a Config object with the .env path
    env_config = Config(RepositoryEnv(env_path))

    # Retrieve the environment variables
    SPOTIPY_CLIENT_ID = env_config('SPOTIPY_CLIENT_ID')
    SPOTIPY_CLIENT_SECRET = env_config('SPOTIPY_CLIENT_SECRET')

    logger.debug('Initializing Spotify client.')
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET
    ))
    return sp


def fetch_playlist_tracks(playlist_id):
    """Fetch all tracks from a Spotify playlist."""

    logger.debug(f'Fetching tracks for playlist ID: {playlist_id}')
    sp = get_spotify_client()
    track_items = []
    
    results = sp.playlist_tracks(playlist_id)
    track_items.extend(results['items'])
    
    # Paginate through all track items
    while results['next']:
        results = sp.next(results)
        track_items.extend(results['items'])
    
    return track_items


def fetch_track_details(track_id):
    """Fetch audio features and additional details for a specific track."""
    sp = get_spotify_client()
    
    audio_features = sp.audio_features(track_id)[0]
    track_info = sp.track(track_id)

    external_urls = track_info.get('external_urls', {})

    return {
        'audio_features': audio_features,
        'track_info': track_info,
        'external_urls': external_urls,
    }


def fetch_playlist_with_details(playlist_id):
    """Fetch all tracks and their details (audio features, popularity) from a playlist."""
    track_items = fetch_playlist_tracks(playlist_id)
    
    detailed_tracks = []
    
    for item in track_items:
        track_data = item['track']
        track_id = track_data['id']
        
        # Get audio features and other details
        details = fetch_track_details(track_id)
        detailed_tracks.append({
            'spotify_id': track_id,
            'name': track_data['name'],
            'album': track_data['album']['name'],
            'artists': ', '.join([artist['name'] for artist in track_data['artists']]),
            'popularity': track_data['popularity'],
            'audio_features': details['audio_features'],
            'external_urls': details['external_urls'],
        })
    
    return detailed_tracks
 