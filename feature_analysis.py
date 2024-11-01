import os
import django
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Django environment (adjust the path as necessary)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from trending.models import Track

# Query all tracks
tracks = Track.objects.all()

# Prepare data for plotting
data = {
    'Popularity': [],
    'Danceability': [],
    'Tempo': [],
    'Valence': [],
    'Speechiness': [],
    'Liveness': []
}

# Fill the data dictionary with track features
for track in tracks:
    data['Popularity'].append(track.popularity)
    data['Danceability'].append(track.danceability)
    data['Tempo'].append(track.tempo)
    data['Valence'].append(track.valence)
    data['Speechiness'].append(track.speechiness)
    data['Liveness'].append(track.liveness)

# Convert to DataFrame for easier plotting
df = pd.DataFrame(data)

# Set the style of the visualization
sns.set(style="whitegrid")

# Create scatter plots
plt.figure(figsize=(12, 10))

# 1st Scatter Plot: Valence vs. Popularity
plt.subplot(3, 2, 1)
sns.scatterplot(data=df, 
                 x='Valence', 
                 y='Popularity', 
                 color='lightblue', 
                 alpha=0.6)
plt.title('Valence vs. Popularity')
plt.xlabel('Valence')
plt.ylabel('Popularity')

# 2nd Scatter Plot: Tempo vs. Popularity
plt.subplot(3, 2, 2)
sns.scatterplot(data=df, 
                 x='Tempo', 
                 y='Popularity', 
                 color='lightblue', 
                 alpha=0.6)
plt.title('Tempo vs. Popularity')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Popularity')

# 3rd Scatter Plot: Danceability vs. Popularity
plt.subplot(3, 2, 3)
sns.scatterplot(data=df, 
                 x='Danceability', 
                 y='Popularity', 
                 color='lightblue', 
                 alpha=0.6)
plt.title('Danceability vs. Popularity')
plt.xlabel('Danceability')
plt.ylabel('Popularity')

# 4th Scatter Plot: Speechiness vs. Popularity
plt.subplot(3, 2, 4)
sns.scatterplot(data=df, 
                 x='Speechiness', 
                 y='Popularity', 
                 color='lightblue', 
                 alpha=0.6)
plt.title('Speechiness vs. Popularity')
plt.xlabel('Speechiness')
plt.ylabel('Popularity')

# 5th Scatter Plot: Liveness vs. Popularity
plt.subplot(3, 2, 5)
sns.scatterplot(data=df, 
                 x='Liveness', 
                 y='Popularity', 
                 color='lightblue', 
                 alpha=0.6)
plt.title('Liveness vs. Popularity')
plt.xlabel('Liveness')
plt.ylabel('Popularity')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
