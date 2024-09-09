import spotify_api

def test_get_spotify_client():
    """Test initializing Spotify client."""
    sp = spotify_api.get_spotify_client()
    print("Spotify client initialized successfully.")
    print("Client:", sp)

def test_fetch_playlist_tracks(playlist_id):
    """Test fetching tracks from a Spotify playlist."""
    tracks = spotify_api.fetch_playlist_tracks(playlist_id)
    print(f"Tracks for playlist ID '{playlist_id}':")
    for track in tracks:
        print(f"- {track['track']['name']} by {', '.join(artist['name'] for artist in track['track']['artists'])}")

def test_fetch_track_details(track_id):
    """Test fetching details for a specific track."""
    details = spotify_api.fetch_track_details(track_id)
    print(f"Details for track ID '{track_id}':")
    print("Audio Features:", details['audio_features'])
    print("Track Info:", details['track_info'])

def test_fetch_playlist_with_details(playlist_id):
    """Test fetching detailed track info for all tracks in a playlist."""
    detailed_tracks = spotify_api.fetch_playlist_with_details(playlist_id)
    print(f"Detailed tracks for playlist ID '{playlist_id}':")
    for track in detailed_tracks:
        print(f"ID: {track['spotify_id']}")
        print(f"Name: {track['name']}")
        print(f"Album: {track['album']}")
        print(f"Artists: {track['artists']}")
        print(f"Popularity: {track['popularity']}")
        print(f"Audio Features: {track['audio_features']}")
        print("-" * 40)

if __name__ == "__main__":
    # Replace with your actual playlist ID and track ID for testing
    test_playlist_id = '37i9dQZF1DXcBWIGoYBM5M'
    test_track_id = '6dOtVTDdiauQNBQEDOtlAB'

    # Test Spotify client
    test_get_spotify_client()
    
    # Test fetching tracks from a playlist
    test_fetch_playlist_tracks(test_playlist_id)
    
    # Test fetching track details
    test_fetch_track_details(test_track_id)
    
    # Test fetching detailed track info for all tracks in a playlist
    test_fetch_playlist_with_details(test_playlist_id)
