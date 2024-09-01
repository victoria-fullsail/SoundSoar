from django.urls import path
from . import views

app_name = 'trending'

urlpatterns = [
    path('', views.trending, name='trending'),  # Default view for /trending
    path('filtered/<str:chart_type>/<str:chart_name>/', views.trending_filtered, name='trending_filtered'),  # Filtered view
    path('review/', views.review, name='review'),
]
