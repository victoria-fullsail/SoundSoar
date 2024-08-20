from django.shortcuts import render

def recommendations(request):
    context = {}
    return render(request, 'personalized/recommendations.html', context)
