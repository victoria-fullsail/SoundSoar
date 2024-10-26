# test_script.py

import os
import django

# Set up Django environment (adjust the path as necessary)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from trending.spotify_search import SpotifySearch
from trending.models import Track, TrackFeatures

def main():
    # Define your search query
    query = "brat"

    # Initialize SpotifySearch instance
    searcher = SpotifySearch()

    # Perform the search
    track_data = searcher.search_tracks(query)

    # Set important fields to None
    rf_predicted_trend = None
    hgb_predicted_trend = None
    lr_predicted_trend = None
    svm_predicted_trend = None
    extra_predicted_trend = None
    lda_predicted_trend = None
    knn_predicted_trend = None
    predicted_trend = None

    # Process each track
    for track in track_data:
        print(f"\nTrack Name: {track['name']}")
        print(f"Artist: {track['artists'][0]['name']}")
        print(f"Album: {track['album']['name']}")
        print(f"Spotify ID: {track['id']}")
        spotify_url = track['external_urls']['spotify']
        print(f"Spotify URL: {spotify_url}")

        # Get Spotify ID and check if it is a training data track
        sp_id = track['id']  # Get Spotify ID
        db_track = Track.objects.filter(spotify_id=sp_id).first()
        
        if db_track:
            print("Found track in database.")
            db_track_features = TrackFeatures.objects.filter(track=db_track).first()
            rf_predicted_trend = db_track_features.rf_prediction
            hgb_predicted_trend = db_track_features.hgb_prediction
            lr_predicted_trend = db_track_features.lr_prediction
            svm_predicted_trend = db_track_features.svm_prediction
            extra_predicted_trend = db_track_features.extra_prediction
            lda_predicted_trend = db_track_features.lda_prediction
            knn_predicted_trend = db_track_features.knn_prediction
            predicted_trend = db_track_features.predicted_trend

            print(f"RF Prediction: {rf_predicted_trend}")
            print(f"HGB Prediction: {hgb_predicted_trend}")
            print(f"LR Prediction: {lr_predicted_trend}")
            print(f"SVM Prediction: {svm_predicted_trend}")
            print(f"Extra Trees Prediction: {extra_predicted_trend}")
            print(f"LDA Prediction: {lda_predicted_trend}")
            print(f"KNN Prediction: {knn_predicted_trend}")
            print(f"Predicted Trend: {predicted_trend}")

        else:
            print("Track not found in database. Making predictions using static method.")
            # Fetch audio features for the track
            audio_features = searcher.get_audio_features(sp_id)
            # Check if audio features were retrieved
            if audio_features is not None:
                print("Audio Features for Track ID:", sp_id)
                for feature, value in audio_features.items():
                    print(f"{feature}: {value}")
            else:
                print(f"No audio features found for Track ID: {sp_id}")
            # Populate the data dictionary with necessary feature values for the prediction
            # Include audio features if they were retrieved
            data = {
                'track__valence': audio_features['valence'] if audio_features else 0.0,
                'track__tempo': audio_features['tempo'] if audio_features else 0,
                'track__speechiness': audio_features['speechiness'] if audio_features else 0.00,
                'track__danceability': audio_features['danceability'] if audio_features else 0.0,
                'track__liveness': audio_features['liveness'] if audio_features else 0.0,
                'velocity': 0.0,
                'current_popularity': track.get('popularity', 0),
                'median_popularity': 0,
                'mean_popularity': 0,
                'std_popularity': 0,
                'retrieval_frequency': 0 
            }

            # Print the data dictionary for debugging
            print("Data for prediction:", data)

            # Make predictions using the static method for each model
            rf_predicted_trend = TrackFeatures.make_active_prediction_no_pop_history("RandomForest", data)
            hgb_predicted_trend = TrackFeatures.make_active_prediction_no_pop_history("HistGradientBoosting", data)
            lr_predicted_trend = TrackFeatures.make_active_prediction_no_pop_history("LogisticRegression", data)
            svm_predicted_trend = TrackFeatures.make_active_prediction_no_pop_history("SVM", data)
            extra_predicted_trend = TrackFeatures.make_active_prediction_no_pop_history("ExtraTrees", data)
            lda_predicted_trend = TrackFeatures.make_active_prediction_no_pop_history("LDA", data)
            knn_predicted_trend = TrackFeatures.make_active_prediction_no_pop_history("KNN", data)

            # Print predictions from static method
            print(f"RF Prediction: {rf_predicted_trend}")
            print(f"HGB Prediction: {hgb_predicted_trend}")
            print(f"LR Prediction: {lr_predicted_trend}")
            print(f"SVM Prediction: {svm_predicted_trend}")
            print(f"Extra Trees Prediction: {extra_predicted_trend}")
            print(f"LDA Prediction: {lda_predicted_trend}")
            print(f"KNN Prediction: {knn_predicted_trend}")

            # Choose the best prediction based on your logic
            predicted_trend = knn_predicted_trend  # Adjust as needed

            print(f"Final Predicted Trend from static method: {predicted_trend}")


if __name__ == "__main__":
    main()
