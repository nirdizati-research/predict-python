import json

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.generics import ListAPIView, GenericAPIView
from rest_framework.mixins import RetrieveModelMixin
from rest_framework.response import Response

from core.constants import CLASSIFICATION, NEXT_ACTIVITY, REGRESSION
from jobs.models import CREATED
from jobs.serializers import JobSerializer
from logs.models import Split
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

    # TODO remove?
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


@api_view(['POST'])
def create_multiple(request):
    """No request validation"""
    payload = json.loads(request.body)

    try:
        split = Split.objects.get(pk=payload['split_id'])
    except Split.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)
    jobs = []

    if payload['type'] == CLASSIFICATION:
        jobs = generate(split, payload)
    elif payload['type'] == NEXT_ACTIVITY:
        jobs = generate(split, payload, NEXT_ACTIVITY)
    elif payload['type'] == REGRESSION:
        jobs = generate(split, payload, REGRESSION)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)


def generate(split, payload, type=CLASSIFICATION):
    jobs = []

    for encoding in payload['config']['encodings']:
        for clustering in payload['config']['clusterings']:
            for method in payload['config']['methods']:
                item = Job.objects.create(
                    split=split,
                    status=CREATED,
                    type=type,
                    config=create_config(payload, encoding, clustering, method))
                jobs.append(item)

    return jobs


def create_config(payload, encoding, clustering, method):
    """Turn lists to single values"""
    config = dict(payload['config'])
    del config['encodings']
    del config['clusterings']
    del config['methods']
    config['encoding'] = encoding
    config['clustering'] = clustering
    config['method'] = method
    return config
