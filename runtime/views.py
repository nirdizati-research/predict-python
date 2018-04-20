import django_rq
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from jobs.models import CREATED
from jobs.serializers import JobSerializer
from predModels.models import PredModels
from jobs.models import Job
from core.constants import NO_CLUSTER
from .replayer import Replayer
from .tasks import runtime_task

@api_view(['GET'])
def get_demo(request, pk):
    
    replay=Replayer(pk)
    replay.start()
    
    return Response("Finito")

@api_view(['GET'])
def get_prediction(request, pk1, pk2, pk3, pk4):
    models=[]
    jobs=[]
    try:
        if not pk2 == 0:
            model = PredModels.objects.get(pk=pk2)
            models.append(model)
        if not pk3 == 0:
            model = PredModels.objects.get(pk=pk3)
            models.append(model)
        if not pk4 == 0:
            model = PredModels.objects.get(pk=pk4)
            models.append(model)
    except PredModels.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)

    for model in models:
        job = generate_run(pk1, model, model.id)
    
    #django_rq.enqueue(training, jobrun, model)
        django_rq.enqueue(runtime_task, job, model)
    #os.system('python3 manage.py rqworker --burst')
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)

def generate_run(logid, model, modelid):
    jobs = []
    split=model.split
    encoding= model.encoding
    if split.type == NO_CLUSTER:
        clustering = 'noCluster'
    else:
        clustering= 'kmeans'
    method= model.method
    item = Job.objects.create(
        status=CREATED,
        type=model.type,
        config=create_config_run(encoding, clustering, method, job.config['padding'], log=logid, model=modelid))
    return item

def create_config_run(encoding, clustering, method, padding, log='', model=''):
    """Turn lists to single values"""
    config = dict()
    config['encoding'] = encoding
    config['clustering'] = clustering
    config['add_label'] = False
    config['padding'] = padding
    config['method'] = method
    config['log_id'] = log
    config['model_id'] = model
    return config