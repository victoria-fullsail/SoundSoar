from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.admin.views.decorators import staff_member_required
from .models import Chart, TrackFeatures

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
        # Default case should not be reached here because `trending` already handles it
        return redirect('trending:trending')
    
    # Filter by the given chart type and name
    filtered_chart = get_object_or_404(Chart, chart_type=chart_type, name=chart_name)

    context = {
        'charts': [filtered_chart]  # Pass the filtered chart in a list to keep the template structure consistent
    }
    return render(request, 'trending/trending.html', context)


@staff_member_required
def review(request):
    # Retrieve a sample TrackFeatures object for display
    track_features = TrackFeatures.objects.order_by('-updated_at').first()  # Fetch the most recent TrackFeatures

    context = {
        'track_features': track_features
    }
    return render(request, 'trending/ready-review.html', context)