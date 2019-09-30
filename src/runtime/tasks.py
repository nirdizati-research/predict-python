from django_rq.decorators import job

from src.core.core import runtime_calculate
from src.jobs.models import JobStatuses
from src.jobs.ws_publisher import publish
from src.logs.models import Log
from src.utils.file_service import get_log


@job("default", timeout='100h')
def runtime_task(job):
    print("Start runtime task ID {}".format(job.pk))
    try:
        job.status = JobStatuses.RUNNING.value
        job.save()
        result = runtime_calculate(job)
        job.result = result
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

