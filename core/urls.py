from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.home, name="home"),
    path('logout/', views.custom_logout, name="custom_logout"),
]