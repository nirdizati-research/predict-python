import django_rq
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from pred_models.models import PredModels
from pred_models.serializers import ModelSerializer
from src.jobs.models import Job, JobStatuses
from src.jobs.serializers import JobSerializer
from src.logs.models import Log
from src.split.models import Split
from .models import XTrace
from .replayer import Replayer
from .serializers import TraceSerializer
from .tasks import runtime_task


@api_view(['GET'])
def tracesList(request):  # TODO: changed self to request, check if correct or not
    traces = XTrace.objects.all()
    serializer = TraceSerializer(traces, many=True)
    return Response(serializer.data, status=200)


@api_view(['GET'])
def modelList(request):
    models = PredModels.objects.all()
    serializer = ModelSerializer(models, many=True)
    return Response(serializer.data, status=200)


@api_view(['GET'])
def get_demo(request, pk, pk1, pk2):
    replay = Replayer(pk, pk1, pk2)
    replay.start()

    return Response("Finito")


@api_view(['GET'])
def get_prediction(request, pk1, pk2, pk3):
    models = []
    jobs = []
    pk1 = int(pk1)
    pk2 = int(pk2)
    pk3 = int(pk3)

    log = Log.objects.get(pk=pk1)
    split, created = Split.objects.get_or_create(type='single', original_log=log)

    try:
        if pk2 > 0:
            model = PredModels.objects.get(pk=pk2)
            models.append(model)
        if pk3 > 0:
            model = PredModels.objects.get(pk=pk3)
            models.append(model)
    except PredModels.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)

    for model in models:
        job = generate_run(pk1, model, model.id, split)

        # django_rq.enqueue(training, jobrun, predictive_model)
        django_rq.enqueue(runtime_task, job, model)
    # os.system('python3 manage.py rqworker --burst')
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)


def generate_run(logid, model, modelid, split):
    config = create_config_run(model.config, log=logid, model=modelid)
    try:
        item = Job.objects.get(split=split, config=config, type=model.type)
    except Job.DoesNotExist:
        item = Job.objects.create(
            status=JobStatuses.CREATED.value,
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
