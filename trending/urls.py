from django.urls import path
from . import views

app_name = 'trending'

urlpatterns = [
    path('', views.trending, name='trending'),
    path('search/', views.search_spotify, name='search_spotify'),
    path('filtered/<str:chart_type>/<str:chart_name>/', views.trending_filtered, name='trending_filtered'),
    path('track/<int:pk>/', views.TrackDetailView.as_view(), name='track_detail'),
    path('review/', views.review, name='review'),
    path('model/info/', views.trend_model_info, name='trend_model_info'),
    path('model/info/randomforest/', views.rf_model_info, name='rf_model_info'),
    path('model/info/histgradientboosting/', views.hgb_model_info, name='hgb_model_info'),
    path('model/info/logisticregression/', views.lr_model_info, name='lr_model_info'),
    path('model/info/supportvectormachine/', views.svm_model_info, name='svm_model_info'),
    path('model/info/lineardiscriminantanalysis/', views.lda_model_info, name='lda_model_info'),
    path('model/info/extratrees/', views.extra_model_info, name='extra_model_info'),
    path('model/info/knearestneighbors/', views.knn_model_info, name='knn_model_info'),
]
