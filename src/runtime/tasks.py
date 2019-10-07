from django_rq.decorators import job

from src.core.core import runtime_calculate, replay_prediction_calculate
from src.jobs.models import JobStatuses
from src.jobs.tasks import prediction_task
from src.jobs.ws_publisher import publish
from src.utils.django_orm import duplicate_orm_row
from .replay import replay_core



@job("default", timeout='100h')
def runtime_task(job):
    print("Start runtime task ID {}".format(job.pk))
    try:
        job.status = JobStatuses.RUNNING.value
        job.save()
        result = runtime_calculate(job, log)
        job.results = {'result': str(result)}
        job.status = JobStatuses.COMPLETED.value
        job.error = ''
    except Exception as e:
        print("error " + str(e.__repr__()))
        job.status = JobStatuses.ERROR.value
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()
        publish(job)\

@job("default", timeout='100h')
def replay_prediction_task(job, log):
    print("Start runtime task ID {}".format(job.pk))
    try:
        job.status = JobStatuses.RUNNING.value
        job.save()
        max_len = max(len(trace) for trace in log)
        if job.encoding.prefix_length != max_len:
            new_job = duplicate_orm_row(job)
            new_job.encoding.prefix_length = max_len
            new_job.save()
            prediction_task(new_job.id)
        result = replay_prediction_calculate(job, log)
        job.results = {'result': str(result)}
        job.status = JobStatuses.COMPLETED.value
        job.error = ''
    except Exception as e:
        print("error " + str(e.__repr__()))
        job.status = JobStatuses.ERROR.value
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()
        publish(job)

@job("default", timeout='100h')
def replay_task(job):
    print("Start replay task ID {}".format(job.pk))
    try:
        job.status = JobStatuses.RUNNING.value
        job.save()
        replay_core(job)
        job.status = JobStatuses.COMPLETED.value
        job.error = ''
    except Exception as e:
        print("error " + str(e.__repr__()))
        job.status = JobStatuses.ERROR.value
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()
        publish(job)

