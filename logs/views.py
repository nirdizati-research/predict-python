from django.http import HttpResponse

def logs(request):
    return HttpResponse("Hello world")
