from django_rq.decorators import job

from core.classification import classifier, classifier_run
from training.models import PredModels
from core.next_activity import next_activity, next_activity_run
from logs.file_service import get_logs
from core.regression import regression, regression_run
from encoders.common import encode_training_logs, encode_run_logs
from jobs.models import Job, JobRun, CREATED, RUNNING, COMPLETED, ERROR
from core.constants import CLASSIFICATION, NEXT_ACTIVITY, REGRESSION

@job("default", timeout='1h')
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
        print("error " + str(e.__repr__()))
        job.status = ERROR
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()        
        
def prediction(job,model):
    #print("Start prediction task ID {}".format(job.pk))
    try:
        if job.status == CREATED:
            job.status = RUNNING
            job.save()
            result = calculate(job.to_dict(),model.to_dict())
            job.result = result
            job.status = COMPLETED
    except Exception as e:
        print("error " + str(e.__repr__()))
        job.status = ERROR
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()

def calculate(job,model):
    """ Main entry method for calculations"""
    #print("Start job {} with {}".format(job['type'], get_run(job)))
    run_log = get_logs(job['log_path'])[0]

    # Python dicts are bad
    if 'prefix_length' in job:
        prefix_length = job['prefix_length']
    else:
        prefix_length = 1

    run_df= encode_run_logs(run_log, job['encoding'], job['type'],
                                       prefix_length=prefix_length)

    if job['type'] == CLASSIFICATION:
        results = classifier_run(run_df, model, job)
    elif job['type'] == REGRESSION:
        results= regression_run(run_df, model, job)
    elif job['type'] == NEXT_ACTIVITY:
        results = next_activity_run(run_df, model, job)
    else:
        raise ValueError("Type not supported", job['type'])
    #print("End job {}, {} . Results {}".format(job['type'], get_run(job), results))
    return results

def getModelList(typed):
    models = PredModels.objects.all()        
    results = models.filter(type = typed)
    return results