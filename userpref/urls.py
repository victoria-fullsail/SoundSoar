from django.urls import path
from . import views

app_name = 'userpref'

urlpatterns = [
    path('', views.preferences, name="preferences"),
]