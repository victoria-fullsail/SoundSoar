from django.urls import path
from . import views

app_name = 'personalized'

urlpatterns = [
    path('', views.recommendations, name="recommendations"),
]