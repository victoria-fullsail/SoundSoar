# spotify_search.py
from .spotify_api import get_spotify_client
import time

class SpotifySearch:
    def __init__(self):
        self.sp = get_spotify_client()

    def search_tracks(self, query, limit=7):
        results = self.sp.search(q=query, type='track', limit=limit)
        return results['tracks']['items']
    
    def get_audio_features(self, track_id):
        """
        Fetch audio features for a specific track by ID.

        Args:
            track_id (str): The ID of the track to fetch audio features for.

        Returns:
            dict: A dictionary containing audio features of the specified track.
                If the track is not found, it returns None.
        """
        # Fetch audio features for the specified track ID
        audio_features = self.sp.audio_features([track_id])
    
        time.sleep(1)

        # Check if audio_features returned any data
        if audio_features and isinstance(audio_features, list) and len(audio_features) > 0:
            return audio_features[0]  # Return the first track's features
        else:
            return None


    def search_albums(self, query, limit=5):
        results = self.sp.search(q=query, type='album', limit=limit)
        return results['albums']['items']

    def search_artists(self, query, limit=5):
        results = self.sp.search(q=query, type='artist', limit=limit)
        return results['artists']['items']

    def search_playlists(self, query, limit=5):
        results = self.sp.search(q=query, type='playlist', limit=limit)
        return results['playlists']['items']


# Example usage
if __name__ == "__main__":
    spotify_search = SpotifySearch()
    
    # Search for tracks with the query "360"
    tracks = spotify_search.search_tracks("360")
    print("Tracks found:", tracks)

    # Check if any tracks were found before attempting to access audio features
    if tracks:
        # Retrieve audio features for the first track found
        audio_features_list = spotify_search.get_audio_features(tracks[0]['id'])
        print("Audio features for the first track:", audio_features_list)
    else:
        print("No tracks found for the query.")

    # Search for an album
    #albums = spotify_search.search_albums("Brat")
    #print("Albums found:", albums)

    # Search for an artist
    #artists = spotify_search.search_artists("Charlie")
   # print("Artists found:", artists)

    # Search for a playlist
    #playlists = spotify_search.search_playlists("Chill Vibes")
   # print("Playlists found:", playlists)
