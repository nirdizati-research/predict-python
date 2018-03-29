import json
import os
from subprocess import call

import django_rq
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.generics import ListAPIView, GenericAPIView
from rest_framework.mixins import RetrieveModelMixin
from rest_framework.response import Response

from core.constants import CLASSIFICATION, NEXT_ACTIVITY, REGRESSION
from training.tr_core import calculate
from jobs.tasks import training
from jobs.models import CREATED
from jobs.serializers import JobSerializer
from logs.models import Split, Log
from training.models import PredModels
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
    payload = json.loads(request.body.decode('utf-8'))

    try:
        split = Split.objects.get(pk=payload['split_id'])
    except Split.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)

    if payload['type'] == CLASSIFICATION:
        jobs = generate(split, payload)
    elif payload['type'] == NEXT_ACTIVITY:
        jobs = generate(split, payload, NEXT_ACTIVITY)
    elif payload['type'] == REGRESSION:
        jobs = generate(split, payload, REGRESSION)
    else:
        return Response({'error': 'type not supported'.format(payload['type'])},
                        status=status.HTTP_422_UNPROCESSABLE_ENTITY)
    for job in jobs:
        django_rq.enqueue(training, job)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)

@api_view(['GET'])
def get_model(request, pk):
    """Get log statistics

    End URL with
    * events for event_by_date
    * resources for resources_by_date
    * executions for event_executions
    """
    try:
        job = Job.objects.get(pk=pk)
    except Log.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)
    
    
    training(job)
    
    return Response({'All the models have been created'})

@api_view(['GET'])
def get_prediction(request, pk1, pk2):
    """Get log statistics

    End URL with
    * events for event_by_date
    * resources for resources_by_date
    * executions for event_executions
    """
    
    try:
        model = PredModels.objects.get(pk=pk2)
    except Log.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)

    job = generate_run(pk1, model, model.id)
    
    #django_rq.enqueue(training, jobrun, model)
    training(job,model)
    #os.system('python3 manage.py rqworker --burst')
    serializer = JobSerializer(job)
    return Response(job.result)

@api_view(['POST'])
def create_prediction(request):
    """No request validation"""
    payload = json.loads(request.body.decode('utf-8'))
    
    try:
        log = Logs.objects.get(pk=payload['log_id'])
        model = PredModel.objects.get(pk=payload['model_id'])
    except Split.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)
    
    if model.type == CLASSIFICATION:
        job = generate_run(log, payload)
    elif model.type == NEXT_ACTIVITY:
        job = generate_run(log, payload, NEXT_ACTIVITY)
    elif model.type == REGRESSION:
        job = generate_run(log, payload, REGRESSION)
    else:
        return Response({'error': 'type not supported'.format(model['type'])},
                        status=status.HTTP_422_UNPROCESSABLE_ENTITY)
    
    training(job,model)
    serializer = JobSerializer(job)
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

def generate_run(logid, model, modelid):
    jobs = []
    split=model.split
    encoding= model.encoding
    if split.type == 'single':
        clustering = 'noCluster'
    else:
        clustering= 'kmeans'
    method= model.method
    item = Job.objects.create(
        status=CREATED,
        type=model.type,
        config=create_config_run(encoding, clustering, method, log=logid, model=modelid))
    return item

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

def create_config_run(encoding, clustering, method, log='', model=''):
    """Turn lists to single values"""
    config = dict()
    config['encoding'] = encoding
    config['clustering'] = clustering
    config['method'] = method
    config['log_id'] = log
    config['model_id'] = model
    return config
