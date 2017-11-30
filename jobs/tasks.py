from logging import ERROR

from core.core import calculate
from jobs.models import Job, CREATED, RUNNING, COMPLETED


#@job("high", timeout='1h')
def prediction_task(job_id):
    print("Start prediction task ID {}".format(job_id))
    job = Job.objects.get(id=job_id)
    try:
        if job.status == CREATED:
            job.status = RUNNING
            job.save()
            result = calculate(job.to_dict())
            job.result = result
            job.status = COMPLETED
    except Exception as e:
        print("error " + str(e))
        job.status = ERROR
        job.error = str(e)
    job.save()
