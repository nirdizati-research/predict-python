import logging

from django_rq.decorators import job
from rest_framework import status

from src.core.core import runtime_calculate, replay_prediction_calculate
from src.encoding.models import Encoding
from src.jobs.models import JobStatuses, JobTypes, Job
from src.jobs.tasks import prediction_task
from src.jobs.ws_publisher import publish
from src.split.models import Split
from src.utils.django_orm import duplicate_orm_row
from .replay import replay_core

logger = logging.getLogger(__name__)


@job("default", timeout='100h')
def runtime_task(job):
    logger.info("Start runtime task ID {}".format(job.id))
    try:
        job.status = JobStatuses.RUNNING.value
        job.save()
        result = runtime_calculate(job)
        job.results = {'result': str(result)}
        job.status = JobStatuses.COMPLETED.value
        job.error = ''
    except Exception as e:
        logger.error(e)
        job.status = JobStatuses.ERROR.value
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()
        publish(job)\



@job("default", timeout='100h')
def replay_prediction_task(replay_prediction_job, training_initial_job, log):
    logger.info("Start runtime task ID {}".format(replay_prediction_job.id))
    try:
        replay_prediction_job.status = JobStatuses.RUNNING.value
        replay_prediction_job.save()
        max_len = max(len(trace) for trace in log)
        if replay_prediction_job.encoding.prefix_length != max_len:
            prediction_job = create_prediction_job(training_initial_job, max_len)
            prediction_task(prediction_job.id)
            prediction_job.refresh_from_db()
            new_replay_prediction_job = duplicate_orm_row(prediction_job)
            new_replay_prediction_job.split = Split.objects.filter(pk=replay_prediction_job.split.id)[0]
            new_replay_prediction_job.type = JobTypes.REPLAY_PREDICT.value
            new_replay_prediction_job.status = JobStatuses.CREATED.value
            replay_prediction_task(new_replay_prediction_job, prediction_job, log)
            return
        result = replay_prediction_calculate(replay_prediction_job, log)
        replay_prediction_job.results = {'result': str(result)}
        replay_prediction_job.status = JobStatuses.COMPLETED.value
        replay_prediction_job.error = ''
    except Exception as e:
        logger.error(e)
        replay_prediction_job.status = JobStatuses.ERROR.value
        replay_prediction_job.error = str(e.__repr__())
        raise e
    finally:
        replay_prediction_job.save()
        publish(replay_prediction_job)


@job("default", timeout='100h')
def replay_task(replay_job, training_initial_job):
    logger.info("Start replay task ID {}".format(replay_job.id))
    requests = list()
    try:
        replay_job.status = JobStatuses.RUNNING.value
        replay_job.error = ''
        replay_job.save()
        requests = replay_core(replay_job, training_initial_job)
        replay_job.status = JobStatuses.COMPLETED.value
        for r in requests:
            if r.status_code != status.HTTP_201_CREATED:
                replay_job.error += [r]
    except Exception as e:
        logger.error(e)
        replay_job.status = JobStatuses.ERROR.value
        replay_job.error += [str(e.__repr__())]
        raise e
    finally:
        replay_job.save()
        publish(replay_job)
        return requests


def create_prediction_job(job: Job, max_len: int) -> Job:
    new_job = duplicate_orm_row(job)
    new_job.type = JobTypes.PREDICTION.value
    new_job.status = JobStatuses.CREATED.value
    new_encoding = duplicate_orm_row(Encoding.objects.filter(pk=job.encoding.id)[0])
    new_encoding.prefix_length = max_len
    new_encoding.save()
    new_job.encoding = new_encoding
    new_job.create_models = True
    new_job.save()
    return new_job
