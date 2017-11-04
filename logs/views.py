from django.http import JsonResponse
from django.views.decorators.http import require_GET
from .models import Log
from .serializers import LogSerializer


@require_GET
def log_list(request):
    logs = Log.objects.all()
    serializer = LogSerializer(logs, many=True)
    return JsonResponse(serializer.data, safe=False)
