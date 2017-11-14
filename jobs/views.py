from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from jobs.serializers import JobSerializer
from .models import Job


@csrf_exempt
@api_view(['GET', 'POST'])
def job_list(request):
    """
    List all jobs, or create a new job.
    """
    if request.method == 'GET':
        snippets = Job.objects.all()
        serializer = JobSerializer(snippets, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = JobSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(['GET'])
def job_detail(request, pk):
    """
    Retrieve job by id.
    """
    try:
        snippet = Job.objects.get(pk=pk)
    except Job.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = JobSerializer(snippet)
        return Response(serializer.data)
