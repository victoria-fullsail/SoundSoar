from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.admin.views.decorators import staff_member_required
from django.views.generic import TemplateView
from .models import Chart, Track, TrackFeatures, TrendModel
from .visualizations import generate_top_tracks_interactive_plot, generate_simple_plot

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

def bokeh_test(request):
    script, div = generate_simple_plot()
    return render(request, 'trending/bokeh-test.html', {'script': script, 'div': div})

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

    # Get Bokeh plot

    bokeh_script, bokeh_div = generate_top_tracks_interactive_plot(track_data)


    context = {
        'chart': filtered_chart,
        'track_data': track_data,
        'all_charts': all_charts,
        'bokeh_script': bokeh_script,
        'bokeh_div': bokeh_div,
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