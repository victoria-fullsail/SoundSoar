from django.shortcuts import render

def preferences(request):
    context = {}
    return render(request, 'userpref/preferences.html', context)
