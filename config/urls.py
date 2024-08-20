from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls', namespace="core")),
    path('trending/', include('trending.urls', namespace="trending")),
    path('analysis/', include('analysis.urls', namespace="analysis")),
    path('personalized/', include('personalized.urls', namespace="personalized")),
    path('user/preferences/', include('userpref.urls', namespace="userpref")),
]

admin.site.site_header = "SoundSoar"
admin.site.site_title = "SoundSoar"
admin.site.index_title = "Welcome to Sound Soar!"

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)