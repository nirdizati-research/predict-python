from django_rq.decorators import job

from core.core import runtime_calculate
from jobs.models import RUNNING, COMPLETED, ERROR
from jobs.ws_publisher import publish
from logs.file_service import get_logs
from logs.models import Log


@job("default", timeout='1h')
def runtime_task(job, model):
    print("Start runtime task ID {}".format(job.pk))
    try:
        job.status = RUNNING
        job.save()
        log = Log.objects.get(pk=job.config['log_id'])
        run_log = get_logs(log.path)[0]
        result_data = runtime_calculate(run_log, model.to_dict())
        result = result_data['prediction']
        job.result = result
        job.status = COMPLETED
        job.error = ''
    except Exception as e:
        print("error " + str(e.__repr__()))
        job.status = ERROR
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()
        publish(job)
