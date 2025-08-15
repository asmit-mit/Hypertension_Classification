from django.shortcuts import HttpResponse, render


# Create your views here.
def homepage(request):
    return HttpResponse("Welcome")
