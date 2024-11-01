import plotly.express as px
import pandas as pd


def generate_top_ten_track_plot(track_data):
    """Generate Top 10 Track Plot Based on Popularity"""
    
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
        'Track Name': track_names,
        'Artist': track_artists,
        'Album': track_albums,
        'Popularity': track_popularities
    })
    
    # Sort by popularity in descending order and select the top 10
    df = df.sort_values(by='Popularity', ascending=False).head(10)
    
    # Reverse y-axis to have the most popular at the top
    df = df.iloc[::-1]
    
    # Create a horizontal bar chart
    fig = px.bar(df, 
                 x='Popularity', 
                 y='Track Name', 
                 orientation='h', 
                hover_data={
                    'Artist': True,
                    'Album': True,
                    'Track Name': False,
                    'Popularity': False
                },
                 title='Top 10 Tracks by Popularity',
                 text='Popularity',
                 color_discrete_sequence=['#84b8d9']
                )

    # Update layout for a more compact and sleek look
    fig.update_layout(
        xaxis_title='Popularity',
        yaxis_title='',  # No need for y-axis title in this case
        height=400,  # Adjust height for a more compact chart
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    # Ensure bars are not too wide and text doesn't overlap
    fig.update_traces(textposition='auto')
    
    # Convert the figure to HTML for rendering in a Django template
    chart_html = fig.to_html(full_html=False)
    
    return chart_html

def generate_track_attribute_plot(attributes):
    """Generate a horizontal bar chart for track attributes."""
    
    # Create a DataFrame from the attributes
    data = pd.DataFrame(list(attributes.items()), columns=['Attribute', 'Value'])
    
    # Create horizontal bar plot using Plotly Express
    fig = px.bar(data, 
                 x='Value', 
                 y='Attribute', 
                 orientation='h', 
                 title='Track Attributes', 
                 text='Value',
                 color_discrete_sequence=['#6FAE65']
                 )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Value',
        yaxis_title='Attribute',
        xaxis=dict(range=[0, 1])
    )

    # Return the figure object
    return fig.to_html(full_html=False)

def generate_track_popularity_trend_plot(dates, popularity_scores):
    """Generate a line plot for track popularity over time."""
    
    # Create a DataFrame from historical data
    data = pd.DataFrame({
        'Date': dates,
        'Popularity': popularity_scores
    })

    # Create line plot using Plotly Express
    fig = px.line(data, 
                  x='Date', 
                  y='Popularity', 
                  title='Track Popularity Over Time', 
                  markers=True, 
                  color_discrete_sequence=['#84b8d9']
    )
    
    # Update layout for the popularity trend plot
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Popularity',
        xaxis=dict(type='date'),
    )

    # Return the figure object
    return fig.to_html(full_html=False)
