import django_rq
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from jobs.models import CREATED
from jobs.serializers import JobSerializer
from predModels.serializers import ModelSerializer
from predModels.models import PredModels
from jobs.models import Job
from logs.models import Log, Split
from core.constants import NO_CLUSTER
from .replayer import Replayer
from .tasks import runtime_task

@api_view(['GET'])
def modelList(request):
    
    models=PredModels.objects.all()
    serializer = ModelSerializer(models, many=True)
    return Response(serializer.data, status=201)

@api_view(['GET'])
def get_demo(request, pk):
    
    replay=Replayer(pk)
    replay.start()
    
    return Response("Finito")

@api_view(['GET'])
def get_prediction(request, pk1, pk2, pk3, pk4):
    models=[]
    jobs=[]
    pk1=int(pk1)
    pk2=int(pk2)
    pk3=int(pk3)
    pk4=int(pk4)
    
    log = Log.objects.get(pk=pk1)
    split, created = Split.objects.get_or_create(type='single', original_log=log)
    
    try:
        if pk2 > 0:
            model = PredModels.objects.get(pk=pk2)
            models.append(model)
        if pk3 > 0:
            model = PredModels.objects.get(pk=pk3)
            models.append(model)
        if pk4 > 0:
            model = PredModels.objects.get(pk=pk4)
            models.append(model)
    except PredModels.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)

    for model in models:
        job = generate_run(pk1, model, model.id, split)
    
    #django_rq.enqueue(training, jobrun, model)
        django_rq.enqueue(runtime_task, job, model)
    #os.system('python3 manage.py rqworker --burst')
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)

def generate_run(logid, model, modelid, split):
    jobs = []
    config = create_config_run(model.config, log=logid, model=modelid)
    try:
        item = Job.objects.get(split=split, config=config, type=model.type)
    except Job.DoesNotExist:
        item = Job.objects.create(
            status=CREATED,
            type=model.type,
            config=config,
            split=split)
    return item

def create_config_run(config, log='', model=''):
    """Turn lists to single values"""
    config = config
    config['add_label'] = False
    config['log_id'] = log
    config['model_id'] = model
    return config