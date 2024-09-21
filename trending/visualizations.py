from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool
import pandas as pd


def generate_simple_plot():
    # Sample data
    data = {
        'x': [1, 2, 3, 4, 5],
        'y': [6, 7, 2, 4, 5],
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create a Bokeh figure
    p = figure(title="Simple Line Plot", x_axis_label='X Axis', y_axis_label='Y Axis')
    p.line(df['x'], df['y'], line_width=2)

    # Return the script and div to embed in the template
    script, div = components(p)
    
    return script, div


def generate_top_tracks_interactive_plot(track_data):
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

    # Prepare the data for Bokeh
    source = ColumnDataSource(df)

    # Create a Bokeh figure
    p = figure(title="Top Tracks by Popularity", 
               x_axis_label='Popularity', 
               y_axis_label='Track Name',
               tools="hover", 
               tooltips="@track_name: @popularity<br>@artist: @album",
               sizing_mode="stretch_both")

    # Add circles to the plot
    p.circle(x='popularity', y='track_name', size=10, source=source)

    # Return the script and div to embed in the template
    script, div = components(p)

    return script, div

def generate_track_detail_interactive_plot(track_data):
    # Implement the function for track details as needed
    pass
