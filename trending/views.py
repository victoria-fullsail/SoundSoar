from django.shortcuts import render

def trending(request):
    context = {}
    return render(request, 'trending/trending.html', context)
