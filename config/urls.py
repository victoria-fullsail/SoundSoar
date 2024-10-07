from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
import debug_toolbar


urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('', include('core.urls', namespace="core")),
    path('trending/', include('trending.urls', namespace="trending")),
    path('popularity/', include('popularity.urls', namespace="popularity")),
    path('personalized/', include('personalized.urls', namespace="personalized")),
    path('user/preferences/', include('userpref.urls', namespace="userpref")),
]

admin.site.site_header = "SoundSoarAdmin"
admin.site.site_title = "SoundSoarAdmin"
admin.site.index_title = "Welcome to Sound Soar!"

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += [path('__debug__/', include(debug_toolbar.urls))]
