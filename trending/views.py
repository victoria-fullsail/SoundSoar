from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import TemplateView
from django.utils import timezone
from .models import Chart, Track, TrackFeatures, TrendModel, FeatureImportance, PopularityHistory, Playlist
from .visualizations import generate_top_ten_track_plot, generate_track_attribute_plot, generate_track_popularity_trend_plot
from .spotify_search import SpotifySearch
from datetime import timedelta

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

    # Active Models - Features and CSV
    feature_importance_active = {}
    for active_model in active_models:
        # Fetch the feature importance tuples
        feature_importance_list = FeatureImportance.objects.filter(trend_model=active_model).values_list('feature_name', 'importance')
        # Create a list of formatted strings for feature importance
        feature_importance_active[active_model.id] = [f"{feature[0]} - {feature[1]:.4f}" for feature in feature_importance_list]

    # Inactive Models - Features and CSV
    feature_importance_inactive = {}
    for inactive_model in inactive_models:
        # Similar processing for inactive models
        feature_importance_list = FeatureImportance.objects.filter(trend_model=inactive_model).values_list('feature_name', 'importance')
        feature_importance_inactive[inactive_model.id] = [f"{feature[0]} - {feature[1]:.4f}" for feature in feature_importance_list]

    context = {
        'active_models': active_models,
        'inactive_models': inactive_models,
        'feature_importance_active': feature_importance_active,
        'feature_importance_inactive': feature_importance_inactive,
    }
    return render(request, 'trending/trend-model.html', context)

def review(request):
    # Query for total number of rows for Track
    total_tracks = Track.objects.count()

    # Retrieve all Spotify playlists (assuming you have a Playlist model)
    spotify_playlists = Playlist.objects.all()

    # Retrieve the most recently updated Track and TrackFeatures
    track = Track.objects.order_by('-updated_at').first()
    track_feature = TrackFeatures.objects.order_by('-updated_at').first()

    # Query for popularity history for the last 30 days for the selected track
    if track:
        thirty_days_ago = timezone.now() - timedelta(days=30)
        popularity_history = PopularityHistory.objects.filter(track=track, timestamp__gte=thirty_days_ago)
    else:
        popularity_history = []  # If no track is available, set to an empty list

    # Get the most recent TrendModel based on creation date
    recent_model = TrendModel.objects.latest('created_at')

    # Get feature importances for the most recent model
    feature_importances = FeatureImportance.objects.filter(trend_model=recent_model).values_list('feature_name', 'importance')

    context = {
        'total_tracks': total_tracks,
        'spotify_playlists': spotify_playlists,
        'track': track,
        'track_feature': track_feature,
        'popularity_history': popularity_history,
        'feature_importances' : feature_importances
    }
    
    return render(request, 'trending/ready-review.html', context)

def rf_model_info(request):
    models = TrendModel.objects.filter(model_type='RandomForest').order_by('-created_at')
    average_accuracy = TrendModel.get_average_score_for_type(model_type='RandomForest', metric='accuracy')
    average_precision = TrendModel.get_average_score_for_type(model_type='RandomForest', metric='precision')
    average_recall = TrendModel.get_average_score_for_type(model_type='RandomForest', metric='recall')
    average_f1_score = TrendModel.get_average_score_for_type(model_type='RandomForest', metric='f1_score')
    best_params = TrendModel.get_parameters_count_list_for_type(model_type='RandomForest')
    context = {
        'models': models,
        'average_accuracy': average_accuracy,
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1_score': average_f1_score,
        'best_params': best_params,
    }
    return render(request, 'trending/rf-model.html', context)

def hgb_model_info(request):
    models = TrendModel.objects.filter(model_type='HistGradientBoosting').order_by('-created_at')
    average_accuracy = TrendModel.get_average_score_for_type(model_type='HistGradientBoosting', metric='accuracy')
    average_precision = TrendModel.get_average_score_for_type(model_type='HistGradientBoosting', metric='precision')
    average_recall = TrendModel.get_average_score_for_type(model_type='HistGradientBoosting', metric='recall')
    average_f1_score = TrendModel.get_average_score_for_type(model_type='HistGradientBoosting', metric='f1_score')
    best_params = TrendModel.get_parameters_count_list_for_type(model_type='HistGradientBoosting')
    context = {
        'models': models,
        'average_accuracy': average_accuracy,
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1_score': average_f1_score,
        'best_params': best_params,
    }
    return render(request, 'trending/hgb-model.html', context)

def lr_model_info(request):
    models = TrendModel.objects.filter(model_type='LogisticRegression').order_by('-created_at')
    average_accuracy = TrendModel.get_average_score_for_type(model_type='LogisticRegression', metric='accuracy')
    average_precision = TrendModel.get_average_score_for_type(model_type='LogisticRegression', metric='precision')
    average_recall = TrendModel.get_average_score_for_type(model_type='LogisticRegression', metric='recall')
    average_f1_score = TrendModel.get_average_score_for_type(model_type='LogisticRegression', metric='f1_score')
    best_params = TrendModel.get_parameters_count_list_for_type(model_type='LogisticRegression')
    context = {
        'models': models,
        'average_accuracy': average_accuracy,
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1_score': average_f1_score,
        'best_params': best_params,
    }
    return render(request, 'trending/lr-model.html', context)

def svm_model_info(request):
    models = TrendModel.objects.filter(model_type='SVM').order_by('-created_at')
    average_accuracy = TrendModel.get_average_score_for_type(model_type='SVM', metric='accuracy')
    average_precision = TrendModel.get_average_score_for_type(model_type='SVM', metric='precision')
    average_recall = TrendModel.get_average_score_for_type(model_type='SVM', metric='recall')
    average_f1_score = TrendModel.get_average_score_for_type(model_type='SVM', metric='f1_score')
    best_params = TrendModel.get_parameters_count_list_for_type(model_type='SVM')
    context = {
        'models': models,
        'average_accuracy': average_accuracy,
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1_score': average_f1_score,
        'best_params': best_params,
    }
    return render(request, 'trending/svm-model.html', context)

def lda_model_info(request):
    models = TrendModel.objects.filter(model_type='LDA').order_by('-created_at')
    average_accuracy = TrendModel.get_average_score_for_type(model_type='LDA', metric='accuracy')
    average_precision = TrendModel.get_average_score_for_type(model_type='LDA', metric='precision')
    average_recall = TrendModel.get_average_score_for_type(model_type='LDA', metric='recall')
    average_f1_score = TrendModel.get_average_score_for_type(model_type='LDA', metric='f1_score')
    best_params = TrendModel.get_parameters_count_list_for_type(model_type='LDA')
    context = {
        'models': models,
        'average_accuracy': average_accuracy,
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1_score': average_f1_score,
        'best_params': best_params,
    }
    return render(request, 'trending/lda-model.html', context)

def extra_model_info(request):
    models = TrendModel.objects.filter(model_type='ExtraTrees').order_by('-created_at')
    average_accuracy = TrendModel.get_average_score_for_type(model_type='ExtraTrees', metric='accuracy')
    average_precision = TrendModel.get_average_score_for_type(model_type='ExtraTrees', metric='precision')
    average_recall = TrendModel.get_average_score_for_type(model_type='ExtraTrees', metric='recall')
    average_f1_score = TrendModel.get_average_score_for_type(model_type='ExtraTrees', metric='f1_score')
    best_params = TrendModel.get_parameters_count_list_for_type(model_type='ExtraTrees')
    context = {
        'models': models,
        'average_accuracy': average_accuracy,
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1_score': average_f1_score,
        'best_params': best_params,
    }
    return render(request, 'trending/extra-model.html', context)

def knn_model_info(request):
    models = TrendModel.objects.filter(model_type='KNN').order_by('-created_at')
    average_accuracy = TrendModel.get_average_score_for_type(model_type='KNN', metric='accuracy')
    average_precision = TrendModel.get_average_score_for_type(model_type='KNN', metric='precision')
    average_recall = TrendModel.get_average_score_for_type(model_type='KNN', metric='recall')
    average_f1_score = TrendModel.get_average_score_for_type(model_type='KNN', metric='f1_score')
    best_params = TrendModel.get_parameters_count_list_for_type(model_type='KNN')
    context = {
        'models': models,
        'average_accuracy': average_accuracy,
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1_score': average_f1_score,
        'best_params': best_params,
    }
    return render(request, 'trending/knn-model.html', context)

def search_spotify(request):
    query = request.GET.get('query', '')
    searcher = SpotifySearch()

    track_data = []


    # Get the basic track data
    track_data = searcher.search_tracks(query)


    context = {
        'track_data': track_data,
        'query': query,
    }

    return render(request, 'trending/search.html', context)
