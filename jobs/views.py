from rest_framework.generics import ListAPIView, GenericAPIView
from rest_framework.mixins import RetrieveModelMixin
from rest_framework.response import Response

from jobs.serializers import JobSerializer
from .models import Job


class JobList(ListAPIView):
    """
    List all jobs, or create a new job.
    """
    serializer_class = JobSerializer

    def get_queryset(self):
        jobs = Job.objects.all()
        type = self.request.query_params.get('type', None)
        status = self.request.query_params.get('status', None)
        if type is not None:
            jobs = jobs.filter(type=type)
        elif type is not None:
            jobs = jobs.filter(status=status)
        return jobs

    def post(self, request):
        serializer = JobSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class JobDetail(RetrieveModelMixin, GenericAPIView):
    queryset = Job.objects.all()
    serializer_class = JobSerializer

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)
