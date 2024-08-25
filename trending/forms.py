from django import forms
from .models import Sound, SocialMediaPlatform

class TopSoundFilterForm(forms.Form):
    """
    Form for filtering TopSound objects by platform and sentiment.
    """
    platform = forms.ModelChoiceField(
        queryset=SocialMediaPlatform.objects.all(),
        required=False,
        empty_label="All Platforms"
    )
    
    sentiment = forms.ChoiceField(
        choices=[('', 'All Emotions')] + [(code, emotion) for code, emotion in Sound.BASIC_EMOTIONS],
        required=False
    )

    def filter_queryset(self, queryset):
        """
        Filter the queryset based on the form's data.
        """
        platform = self.cleaned_data.get('platform')
        sentiment = self.cleaned_data.get('sentiment')

        if platform:
            queryset = queryset.filter(platform=platform)

        if sentiment:
            queryset = queryset.filter(sentiment=sentiment)

        return queryset
