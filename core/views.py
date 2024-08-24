from django.shortcuts import render
from django.contrib.auth import logout
from django.shortcuts import redirect, get_object_or_404


def home(request):
    context = {}
    return render(request, 'core/index.html', context)

def custom_logout(request):
    logout(request)
    return redirect('core:home')