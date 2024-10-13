# spotify_search.py
from .spotify_api import get_spotify_client

class SpotifySearch:
    def __init__(self):
        self.sp = get_spotify_client()

    def search_tracks(self, query, limit=10):
        results = self.sp.search(q=query, type='track', limit=limit)
        return results['tracks']['items']

    def search_albums(self, query, limit=10):
        results = self.sp.search(q=query, type='album', limit=limit)
        return results['albums']['items']

    def search_artists(self, query, limit=10):
        results = self.sp.search(q=query, type='artist', limit=limit)
        return results['artists']['items']

    def search_playlists(self, query, limit=10):
        results = self.sp.search(q=query, type='playlist', limit=limit)
        return results['playlists']['items']


# Example usage
if __name__ == "__main__":
    spotify_search = SpotifySearch()
    
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
