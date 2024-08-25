from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
from .forms import TopSoundFilterForm
from .models import Sound

@staff_member_required
def trending(request):
    form = TopSoundFilterForm(request.GET or None)
    top_sounds = Sound.objects.all()

    if form.is_valid():
        top_sounds = form.filter_queryset(top_sounds)

    # Prepare data for the table
    headers = ['Platform', 'Sentiment', 'Sound Name', 'Sound URL', 'Usage Count', 'Last Updated']
    data = [
        [
            sound.platform.name,
            sound.get_sentiment_display(),
            sound.sound_name,
            sound.sound_url or 'N/A',
            sound.usage_count,
            sound.last_updated.strftime('%Y-%m-%d %H:%M:%S')
        ]
        for sound in top_sounds
    ]

    context = {
        'form': form,
        'headers': headers,
        'data': data,
    }
    return render(request, 'trending/trending.html', context)
