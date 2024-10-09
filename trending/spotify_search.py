import spotipy
from spotipy.oauth2 import SpotifyOAuth

class SpotifySearch:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                             client_secret=client_secret,
                                                             redirect_uri=redirect_uri,
                                                             scope='user-library-read'))

    def search_tracks(self, query, limit=10):
        """
        Search for tracks matching the query.

        Args:
            query (str): The search query string.
            limit (int): The number of results to return.

        Returns:
            list: A list of track objects.
        """
        results = self.sp.search(q=query, type='track', limit=limit)
        return results['tracks']['items']

    def search_albums(self, query, limit=10):
        """
        Search for albums matching the query.

        Args:
            query (str): The search query string.
            limit (int): The number of results to return.

        Returns:
            list: A list of album objects.
        """
        results = self.sp.search(q=query, type='album', limit=limit)
        return results['albums']['items']

    def search_artists(self, query, limit=10):
        """
        Search for artists matching the query.

        Args:
            query (str): The search query string.
            limit (int): The number of results to return.

        Returns:
            list: A list of artist objects.
        """
        results = self.sp.search(q=query, type='artist', limit=limit)
        return results['artists']['items']

    def search_playlists(self, query, limit=10):
        """
        Search for playlists matching the query.

        Args:
            query (str): The search query string.
            limit (int): The number of results to return.

        Returns:
            list: A list of playlist objects.
        """
        results = self.sp.search(q=query, type='playlist', limit=limit)
        return results['playlists']['items']

# Example usage
if __name__ == "__main__":
    # Replace with your Spotify API credentials
    CLIENT_ID = 'your_client_id'
    CLIENT_SECRET = 'your_client_secret'
    REDIRECT_URI = 'your_redirect_uri'

    spotify_search = SpotifySearch(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
    
    # Search for a track
    tracks = spotify_search.search_tracks("360")
    print("Tracks found:", tracks)

    # Search for an album
    albums = spotify_search.search_albums("Brat")
    print("Albums found:", albums)

    # Search for an artist
    artists = spotify_search.search_artists("Charlie")
    print("Artists found:", artists)

    # Search for a playlist
    playlists = spotify_search.search_playlists("Chill Vibes")
    print("Playlists found:", playlists)
