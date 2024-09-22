from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.admin.views.decorators import staff_member_required
from django.views.generic import TemplateView
from .models import Chart, Track, TrackFeatures, TrendModel
from .visualizations import generate_top_tracks_interactive_plot
import plotly.express as px
from django.shortcuts import render

class TrackDetailView(TemplateView):
    template_name = 'trending/track-detail.html'  # Create this template

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        track_id = self.kwargs['pk']
        track = Track.objects.get(pk=track_id)
        features = TrackFeatures.objects.filter(track=track)  # Adjust based on your relationship

        context['track'] = track
        context['features'] = features
        return context

def chart_view(request):
    # Example data using Plotly's built-in datasets
    df = px.data.gapminder().query("year == 2007")
    
    # Create a scatter plot using Plotly Express
    fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                     hover_name="country", log_x=True, size_max=60)
    
    # Convert the figure to HTML
    chart_html = fig.to_html(full_html=False)

    # Render the template and pass the plot as context
    return render(request, 'trending/plotly-test.html', {'chart': chart_html})

@staff_member_required
def trending(request):
    # Fetch the most recent Spotify playlist
    most_recent_spotify_playlist = Chart.objects.filter(chart_type='spotify_playlist').order_by('-created_at').first()
    
    if most_recent_spotify_playlist:
        # Redirect to the trending_filtered view with the most recent Spotify playlist parameters
        return redirect('trending:trending_filtered', chart_type='spotify_playlist', chart_name=most_recent_spotify_playlist.name)
    else:
        # Handle the case where no Spotify playlists are available
        return render(request, 'trending/trending.html', {'charts': [], 'message': 'No Spotify playlists available.'})


@staff_member_required
def trending_filtered(request, chart_type='spotify_playlist', chart_name=''):
    if chart_type == 'spotify_playlist' and not chart_name:
        return redirect('trending:trending')

    filtered_chart = get_object_or_404(Chart, chart_type=chart_type, name=chart_name)

    # Fetch the first playlist associated with the filtered chart
    playlist = filtered_chart.playlists.first()  # Get the first playlist, adjust as necessary

    if not playlist:
        return render(request, 'trending/trending.html', {'chart': filtered_chart, 'tracks': [], 'message': 'No tracks available for this playlist.'})

    # Fetch all tracks associated with the playlist
    tracks = playlist.tracks.all()

    # Fetch features for those tracks
    track_features = TrackFeatures.objects.filter(track__in=tracks)

     # Create a list of tuples (track, features)
    track_data = []
    for track in tracks:
        feature = track_features.filter(track=track).first()
        track_data.append((track, feature)) 
    
    # Fetch all charts for dropdown
    all_charts = Chart.objects.all()

    # Get Plotly Chart HTML
    topten_chart_html = generate_top_tracks_interactive_plot(track_data)

    context = {
        'chart': filtered_chart,
        'track_data': track_data,
        'all_charts': all_charts,
        'topten_chart_html': topten_chart_html,
    }
    return render(request, 'trending/trending.html', context)


def trend_model_info(request):
    active_model = TrendModel.objects.filter(is_active=True).first()

    context = {
        'active_model': active_model
    }
    return render(request, 'trending/trend-model.html', context)


def review(request):
    # Retrieve a sample TrackFeatures object for display
    track_features = TrackFeatures.objects.order_by('-updated_at').first()  # Fetch the most recent TrackFeatures

    context = {
        'track_features': track_features
    }
    return render(request, 'trending/ready-review.html', context)