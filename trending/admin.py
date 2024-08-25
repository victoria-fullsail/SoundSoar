from django.contrib import admin
from .models import SocialMediaPlatform

@admin.register(SocialMediaPlatform)
class SocialMediaPlatformAdmin(admin.ModelAdmin):
    list_display = ('name', 'api_endpoint', 'api_key')
    search_fields = ('name', 'api_endpoint')
