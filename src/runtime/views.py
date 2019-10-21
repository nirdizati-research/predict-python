import django_rq
from pm4py.objects.log.importer.xes.factory import import_log_from_string
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from src.jobs.models import Job, JobTypes, JobStatuses
from src.jobs.serializers import JobSerializer
from src.split.models import Split
from src.utils.django_orm import duplicate_orm_row
from .tasks import runtime_task, replay_task, replay_prediction_task


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
        new_job.status = JobStatuses.CREATED.value
        new_job.split = split
        new_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'Job ' + str(modelId) + ' not in database'}, status=status.HTTP_404_NOT_FOUND)

    django_rq.enqueue(runtime_task, new_job)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)


@api_view(['POST'])
def post_replay_prediction(request):
    jobs = []
    data = request.data
    modelId = int(data['modelId'])
    training_initial_job_id = int(data['training_job'])

    try:
        training_initial_job = Job.objects.get(pk=training_initial_job_id)
        replay_job = Job.objects.get(pk=modelId)
        replay_prediction_job = duplicate_orm_row(replay_job)
        replay_prediction_job.type = JobTypes.REPLAY_PREDICT.value
        replay_prediction_job.status = JobStatuses.CREATED.value
        replay_prediction_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'Job ' + str(modelId) + ' not in database'}, status=status.HTTP_404_NOT_FOUND)

    log = import_log_from_string(data['log'])
    django_rq.enqueue(replay_prediction_task, replay_prediction_job, training_initial_job,  log)
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
        training_initial_job = Job.objects.get(pk=modelId)
        new_job = duplicate_orm_row(training_initial_job)
        new_job.type = JobTypes.REPLAY.value
        new_job.status = JobStatuses.CREATED.value
        new_job.split = split
        new_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'Job ' + str(modelId) + ' not in database'}, status=status.HTTP_404_NOT_FOUND)

    django_rq.enqueue(replay_task, new_job, training_initial_job)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)


