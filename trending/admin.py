from django.contrib import admin
from .models import Chart, Playlist, Track, PlaylistTrack, PopularityHistory, TrackFeatures

admin.site.register(Chart)
admin.site.register(Playlist)
admin.site.register(Track)
admin.site.register(PlaylistTrack)
admin.site.register(PopularityHistory)
admin.site.register(TrackFeatures)
