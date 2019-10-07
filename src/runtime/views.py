import django_rq
from pm4py.objects.log.importer.xes.factory import import_log_from_string
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from src.jobs.models import Job, JobStatuses, JobTypes
from src.jobs.serializers import JobSerializer
from src.split.models import Split
from src.utils.django_orm import duplicate_orm_row
from .tasks import runtime_task, replay_task, replay_prediction_task

'''
@api_view(['GET'])
def tracesList(request):  # TODO: changed self to request, check if correct or not
    traces = XTrace.objects.all()
    serializer = TraceSerializer(traces, many=True)
    return Response(serializer.data, status=200)


@api_view(['GET'])
def modelList(request):
    completed_jobs = Job.objects.filter(
        status=JobStatuses.COMPLETED.value,
        type=JobTypes.PREDICTION.value,
        create_models=True
    )
    serializer = JobSerializer(completed_jobs, many=True)
    return Response(serializer.data, status=200)


@api_view(['GET'])
def get_demo(request, pk, pk1, pk2):
    replay = Replayer(pk, pk1, pk2)
    replay.start()

    return Response("Finito")

'''
@api_view(['POST'])
def post_prediction(request):
    jobs = []
    data = request.data
    modelId = int(data['modelId'])
    splitId = int(data['splitId'])
    split = Split.objects.get(pk=splitId)

    try:
        job = Job.objects.get(pk=modelId)
        new_job = duplicate_orm_row(job)
        new_job.type = JobTypes.RUNTIME.value
        new_job.spplit = split
        new_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)

    django_rq.enqueue(runtime_task, new_job)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)

@api_view(['POST'])
def post_replay_prediction(request):
    jobs = []
    data = request.data
    modelId = int(data['modelId'])

    try:
        job = Job.objects.get(pk=modelId)
        new_job = duplicate_orm_row(job)
        new_job.type = JobTypes.PREDICT.value
        new_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)

    log = import_log_from_string(data['log'])
    django_rq.enqueue(replay_prediction_task, job, log)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)


@api_view(['POST'])
def post_replay(request):
    jobs = []
    data = request.data
    splitId = int(data['splitId'])
    modelId = int(data['modelId'])

    split = Split.objects.get(pk=splitId)

    try:
        job = Job.objects.get(pk=modelId)
        new_job = duplicate_orm_row(job)
        new_job.type = JobTypes.REPLAY.value
        new_job.split = split
        new_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)


    django_rq.enqueue(replay_task, new_job)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)


