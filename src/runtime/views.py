import logging
import django_rq
from pm4py.objects.log.importer.xes.factory import import_log_from_string
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response

from src.jobs.models import Job, JobTypes, JobStatuses
from src.jobs.serializers import JobSerializer
from src.runtime.tasks import runtime_task, replay_prediction_task, replay_task
from src.split.models import Split
from src.utils.custom_parser import CustomXMLParser
from src.utils.django_orm import duplicate_orm_row

logger = logging.getLogger(__name__)


@api_view(['POST'])
def post_prediction(request):
    jobs = []
    data = request.data
    job_id = int(data['jobId'])
    split_id = int(data['splitId'])
    split = Split.objects.get(pk=split_id)

    try:
        job = Job.objects.get(pk=job_id)
        new_job = duplicate_orm_row(job)
        new_job.type = JobTypes.RUNTIME.value
        new_job.status = JobStatuses.CREATED.value
        new_job.split = split
        new_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'Job ' + str(job_id) + ' not in database'}, status=status.HTTP_404_NOT_FOUND)

    django_rq.enqueue(runtime_task, new_job)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@parser_classes([CustomXMLParser])
def post_replay_prediction(request):
    jobs = []
    job_id = int(request.query_params['jobId'])
    training_initial_job_id = int(request.query_params['training_job'])
    logger.info("Creating replay_prediction task")

    try:
        training_initial_job = Job.objects.get(pk=training_initial_job_id)
        replay_job = Job.objects.filter(pk=job_id)[0]
        replay_prediction_job = duplicate_orm_row(replay_job)
        replay_prediction_job.parent_job = Job.objects.filter(pk=job_id)[0]
        replay_prediction_job.type = JobTypes.REPLAY_PREDICT.value
        replay_prediction_job.status = JobStatuses.CREATED.value
        replay_prediction_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'Job ' + str(job_id) + ' not in database'}, status=status.HTTP_404_NOT_FOUND)

    logger.info("Enqueuing replay_prediction task ID {}".format(replay_prediction_job.id))
    log = import_log_from_string(request.data.decode('utf-8'))
    django_rq.enqueue(replay_prediction_task, replay_prediction_job, training_initial_job,  log)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def post_replay(request):
    jobs = []
    data = request.data
    split_id = int(data['splitId'])
    job_id = int(data['jobId'])

    split = Split.objects.get(pk=split_id)

    try:
        training_initial_job = Job.objects.get(pk=job_id)
        new_job = duplicate_orm_row(training_initial_job)
        new_job.type = JobTypes.REPLAY.value
        new_job.status = JobStatuses.CREATED.value
        new_job.split = split
        new_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'Job ' + str(job_id) + ' not in database'}, status=status.HTTP_404_NOT_FOUND)

    django_rq.enqueue(replay_task, new_job, training_initial_job)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


