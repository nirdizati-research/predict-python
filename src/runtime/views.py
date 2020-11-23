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
    """ Post request to have a single static prediction

        :param request: json
        :return: Response
    """
    jobs = []
    data = request.data
    job_id = int(data['jobId'])
    split_id = int(data['splitId'])
    split = Split.objects.get(pk=split_id)

    try:
        job = Job.objects.get(pk=job_id)
        # new_job = duplicate_orm_row(job)  #todo: replace with simple CREATE
        new_job = Job.objects.create(
            created_date=job.created_date,
            modified_date=job.modified_date,
            error=job.error,
            status=job.status,
            type=job.type,
            create_models=job.create_models,
            case_id=job.case_id,
            event_number=job.event_number,
            gold_value=job.gold_value,
            results=job.results,
            parent_job=job.parent_job,
            split=job.split,
            encoding=job.encoding,
            labelling=job.labelling,
            clustering=job.clustering,
            predictive_model=job.predictive_model,
            evaluation=job.evaluation,
            hyperparameter_optimizer=job.hyperparameter_optimizer,
            incremental_train=job.incremental_train
        )
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
    """ Post request to have a single prediction during the replay of a log

        :param request: json
        :return: Response
    """
    jobs = []
    job_id = int(request.query_params['jobId'])
    training_initial_job_id = int(request.query_params['training_job'])
    logger.info("Creating replay_prediction task")

    try:
        training_initial_job = Job.objects.get(pk=training_initial_job_id)
        replay_job = Job.objects.filter(pk=job_id)[0]
        # replay_prediction_job = duplicate_orm_row(replay_job)  #todo: replace with simple CREATE
        replay_prediction_job = Job.objects.create(
            created_date=replay_job.created_date,
            modified_date=replay_job.modified_date,
            error=replay_job.error,
            status=replay_job.status,
            type=replay_job.type,
            create_models=replay_job.create_models,
            case_id=replay_job.case_id,
            event_number=replay_job.event_number,
            gold_value=replay_job.gold_value,
            results=replay_job.results,
            parent_job=replay_job.parent_job,
            split=replay_job.split,
            encoding=replay_job.encoding,
            labelling=replay_job.labelling,
            clustering=replay_job.clustering,
            predictive_model=replay_job.predictive_model,
            evaluation=replay_job.evaluation,
            hyperparameter_optimizer=replay_job.hyperparameter_optimizer,
            incremental_train=replay_job.incremental_train
        )
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
    """ Post request to start a demo of a log arriving to server

        :param request: json
        :return: Response
    """
    jobs = []
    data = request.data
    split_id = int(data['splitId'])
    job_id = int(data['jobId'])

    split = Split.objects.get(pk=split_id)

    try:
        training_initial_job = Job.objects.get(pk=job_id)
        # new_job = duplicate_orm_row(training_initial_job)  #todo: replace with simple CREATE
        new_job = Job.objects.create(
            created_date=training_initial_job.created_date,
            modified_date=training_initial_job.modified_date,
            error=training_initial_job.error,
            status=training_initial_job.status,
            type=training_initial_job.type,
            create_models=training_initial_job.create_models,
            case_id=training_initial_job.case_id,
            event_number=training_initial_job.event_number,
            gold_value=training_initial_job.gold_value,
            results=training_initial_job.results,
            parent_job=training_initial_job.parent_job,
            split=training_initial_job.split,
            encoding=training_initial_job.encoding,
            labelling=training_initial_job.labelling,
            clustering=training_initial_job.clustering,
            predictive_model=training_initial_job.predictive_model,
            evaluation=training_initial_job.evaluation,
            hyperparameter_optimizer=training_initial_job.hyperparameter_optimizer,
            incremental_train=training_initial_job.incremental_train
        )
        new_job.type = JobTypes.REPLAY.value
        new_job.status = JobStatuses.CREATED.value
        new_job.split = split
        new_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'Job ' + str(job_id) + ' not in database'}, status=status.HTTP_404_NOT_FOUND)

    django_rq.enqueue(replay_task, new_job, training_initial_job)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def get_prediction(request, pk, explanation_target):
    """ Post request to start a demo of a log arriving to server

        :param pk:
        :param explanation_target:
        :param request: json
        :return: Response
    """
    try:
        training_initial_job = Job.objects.get(pk=pk)
        # new_job = duplicate_orm_row(training_initial_job)  #todo: replace with simple CREATE
        new_job = Job.objects.create(
            created_date=training_initial_job.created_date,
            modified_date=training_initial_job.modified_date,
            error=training_initial_job.error,
            status=training_initial_job.status,
            type=training_initial_job.type,
            create_models=training_initial_job.create_models,
            case_id=training_initial_job.case_id,
            event_number=training_initial_job.event_number,
            gold_value=training_initial_job.gold_value,
            results=training_initial_job.results,
            parent_job=training_initial_job.parent_job,
            split=training_initial_job.split,
            encoding=training_initial_job.encoding,
            labelling=training_initial_job.labelling,
            clustering=training_initial_job.clustering,
            predictive_model=training_initial_job.predictive_model,
            evaluation=training_initial_job.evaluation,
            hyperparameter_optimizer=training_initial_job.hyperparameter_optimizer,
            incremental_train=training_initial_job.incremental_train
        )
        new_job.type = JobTypes.REPLAY.value
        new_job.status = JobStatuses.CREATED.value
        new_job.save()
    except Job.DoesNotExist:
        return Response({'error': 'Job ' + str(pk) + ' not in database'}, status=status.HTTP_404_NOT_FOUND)
    return Response(replay_predictions(new_job, Job.objects.get(pk=pk), explanation_target), status=status.HTTP_200_OK)

