from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import TemplateView
from .models import Chart, Track, TrackFeatures, TrendModel
from .visualizations import generate_top_ten_track_plot, generate_track_attribute_plot, generate_track_popularity_trend_plot
from django.shortcuts import render
from .spotify_search import SpotifySearch


class TrackDetailView(TemplateView):
    template_name = 'trending/track-detail.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        track_id = self.kwargs['pk']
        track = Track.objects.get(pk=track_id)
        feature = TrackFeatures.objects.get(track=track)

        # Prepare data for the attributes chart
        attributes = {
            'Danceability': track.danceability,
            'Energy': track.energy,
            'Tempo': track.tempo,
            'Valence': track.valence,
            'Speechiness': track.speechiness,
            'Acousticness': track.acousticness,
            'Instrumentalness': track.instrumentalness,
            'Liveness': track.liveness,
        }

        # Obtain graph obj
        attribute_graph = generate_track_attribute_plot(attributes)

        # Get population data from the last 30 days
        historical_data = feature.get_historical_popularity_tuples()
        dates, popularity_scores = zip(*historical_data) if historical_data else ([], [])

        # Obtain graph obj
        popularity_graph = generate_track_popularity_trend_plot(dates, popularity_scores)

        context['track'] = track
        context['attribute_graph'] = attribute_graph
        context['feature'] = feature
        context['popularity_graph'] = popularity_graph

        return context


def trending(request):
    # Fetch the most recent spotify chart
    most_recent_chart = Chart.objects.filter(chart_type='spotify_playlist').order_by('created_at').first()

    if most_recent_chart:
        # Redirect to the trending_filtered view based on the chart type
        return redirect('trending:trending_filtered', chart_type=most_recent_chart.chart_type, chart_name=most_recent_chart.name)
    else:
        # Handle the case where no playlists are available
        return render(request, 'trending/trending.html', {'charts': [], 'message': 'No playlists available.'})


def trending_filtered(request, chart_type='spotify_playlist', chart_name=''):
    # Fetch the chart based on chart_type and chart_name
    filtered_chart = get_object_or_404(Chart, chart_type=chart_type, name=chart_name)

    # Initialize variables for playlist and track data
    playlist = None
    tracks = None
    track_data = []

    # If chart type is spotify_playlist, get Playlist model data
    if chart_type == 'spotify_playlist':
        playlist = filtered_chart.playlists.first()  # Get the first associated playlist
        if playlist:
            tracks = playlist.tracks.all()

    # If chart type is custom, get CustomPlaylist model data
    elif chart_type == 'custom':
        playlist = filtered_chart.custom_playlists.first()  # Get the first associated custom playlist
        if playlist:
            tracks = playlist.tracks.all()

    # Handle the case where no playlist is available
    if not playlist:
        return render(request, 'trending/trending.html', {'chart': filtered_chart, 'tracks': [], 'message': 'No tracks available for this playlist.'})

    # Fetch track features for the tracks
    track_features = TrackFeatures.objects.filter(track__in=tracks)

    # Create a list of tuples (track, features)
    for track in tracks:
        feature = track_features.filter(track=track).first()
        track_data.append((track, feature))

    # Fetch all charts for the dropdown menu
    all_charts = Chart.objects.all()

    # Generate the Plotly chart (assuming a separate function handles this)
    topten_chart_html = generate_top_ten_track_plot(track_data)

    # Prepare the context
    context = {
        'chart': filtered_chart,
        'track_data': track_data,
        'all_charts': all_charts,
        'topten_chart_html': topten_chart_html,
    }

    return render(request, 'trending/trending.html', context)


def trend_model_info(request):
    active_models = TrendModel.objects.filter(is_active=True).order_by('-created_at')
    inactive_models = TrendModel.objects.filter(is_active=False).order_by('-created_at')
    context = {
        'active_models': active_models,
        'inactive_models': inactive_models,
    }
    return render(request, 'trending/trend-model.html', context)


def review(request):
    # Retrieve a sample TrackFeatures object for display
    track_features = TrackFeatures.objects.order_by('-updated_at').first()  # Fetch the most recent TrackFeatures
    tracks = Track.objects.order_by('-updated_at').first()  # Fetch the most recent TrackFeatures

    context = {
        'track_features': track_features,
        'tracks': tracks
    }
    return render(request, 'trending/ready-review.html', context)


def search_spotify(request):
    query = request.GET.get('query', '')
    searcher = SpotifySearch()

    track_data = []  # Initialize an empty list for track data

    if query:
        # Perform the search for tracks using the SpotifySearch class
        results = searcher.search_tracks(query)
        track_data = results  # Assign the search results directly to track_data

    # Create a context dictionary to pass data to the template
    context = {
        'track_data': track_data,  # Pass the track data to the template
        'query': query,
    }

    return render(request, 'trending/search.html', context)