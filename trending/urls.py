from django.urls import path
from . import views

app_name = 'trending'

urlpatterns = [
    path('', views.trending, name='trending'),
    path('filtered/<str:chart_type>/<str:chart_name>/', views.trending_filtered, name='trending_filtered'),  # Filtered view
    path('track/<int:pk>/', views.TrackDetailView.as_view(), name='track_detail'),
    path('review/', views.review, name='review'),
    path('model/info/', views.trend_model_info, name='trend_model_info'),
]
