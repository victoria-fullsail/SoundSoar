import plotly.express as px
import pandas as pd


def generate_top_tracks_interactive_plot(track_data):
    """Top Ten Track Based on Populartity"""

    # Extract track names, artists, albums, and popularity into separate lists
    track_names = []
    track_artists = []
    track_albums = []
    track_popularities = []

    for track, feature in track_data:
        track_names.append(track.name)
        track_artists.append(track.artist)
        track_albums.append(track.album)
        track_popularities.append(track.popularity)

    # Create a DataFrame
    df = pd.DataFrame({
        'track_name': track_names,
        'artist': track_artists,
        'album': track_albums,
        'popularity': track_popularities,
    })

    # Sort the DataFrame by popularity and keep the top 10
    df = df.sort_values(by='popularity', ascending=False).head(10)

    # Now create the chart, for example, using a horizontal bar chart
    fig = px.bar(df, x='popularity', y='track_name', orientation='h', hover_data=['artist', 'album'], title='Top 10 Tracks by Popularity')

    # Convert the figure to HTML for rendering in a Django template
    chart_html = fig.to_html(full_html=False)

    return chart_html


def generate_track_detail_interactive_plot(track_data):
    # Implement the function for track details as needed
    pass
