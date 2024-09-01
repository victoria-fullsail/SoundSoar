from django.utils import timezone
from trending.models import Chart, Playlist, Track, PlaylistTrack, StreamHistory, TrackFeatures

# Create a Chart
chart = Chart.objects.create(
    name="Top Hits",
    description="Top hits playlist",
    chart_type="spotify_playlist"
)

# Create a Playlist associated with the Chart
playlist = Playlist.objects.create(
    chart=chart,
    playlist_id="37i9dQZF1DXcBWIGoYBM5M",
    name="Today's Top Hits",
    description="The biggest hits of today."
)

# Create some Tracks
track1 = Track.objects.create(
    spotify_id="3n3Ppam7vgaVa1iaRUc9Lp",
    name="Blinding Lights",
    album="After Hours",
    artist="The Weeknd",
    popularity=98,
    genre="Pop",
    danceability=0.53,
    energy=0.73,
    tempo=171.005,
    valence=0.33,
    speechiness=0.059,
    acousticness=0.00146,
    instrumentalness=0.00000,
    liveness=0.0902,
    streams=500000000,
    added_to_playlists_count=10
)

track2 = Track.objects.create(
    spotify_id="7lPN2DXiMsVn7XUKtOW1CS",
    name="Good 4 U",
    album="SOUR",
    artist="Olivia Rodrigo",
    popularity=95,
    genre="Pop",
    danceability=0.53,
    energy=0.80,
    tempo=169.928,
    valence=0.56,
    speechiness=0.0996,
    acousticness=0.0331,
    instrumentalness=0.00000,
    liveness=0.33,
    streams=300000000,
    added_to_playlists_count=5
)


# Add tracks to the Playlist
PlaylistTrack.objects.create(
    playlist=playlist,
    track=track1
)

PlaylistTrack.objects.create(
    playlist=playlist,
    track=track2
)

# Create StreamHistory for Tracks
StreamHistory.objects.create(
    track=track1,
    streams=500000000
)

StreamHistory.objects.create(
    track=track2,
    streams=300000000
)

# Create TrackFeatures for Tracks
TrackFeatures.objects.create(
    track=track1,
    danceability=track1.danceability,
    energy=track1.energy,
    tempo=track1.tempo,
    current_streams=track1.streams,
    streams_last_24h=1000000,
    streams_last_7d=7000000,
    streams_last_30d=30000000,
    current_popularity=track1.popularity,
    velocity=0.5,
    trend='up'
)

TrackFeatures.objects.create(
    track=track2,
    danceability=track2.danceability,
    energy=track2.energy,
    tempo=track2.tempo,
    current_streams=track2.streams,
    streams_last_24h=500000,
    streams_last_7d=3500000,
    streams_last_30d=15000000,
    current_popularity=track2.popularity,
    velocity=0.2,
    trend='stable'
)

