from django_rq.decorators import job

from core.classification import classifier, classifier_run
from training.models import PredModels
from training.tr_core import calculate as tr_calculate
from core.next_activity import next_activity, next_activity_run
from logs.models import Log, Split
from logs.file_service import get_logs
from core.regression import regression, regression_run
from encoders.common import encode_training_logs, encode_run_logs
from jobs.models import Job, JobRun, CREATED, RUNNING, COMPLETED, ERROR
from core.constants import CLASSIFICATION, NEXT_ACTIVITY, REGRESSION

@job("default", timeout='1h')
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
    tr_log = Log.objects.get(name=model['log_name'],path=model['log_path'])
    # Python dicts are bad
    run_df, prefix_length= encode_run_logs(run_log, model['encoding'], model['type'])

    try:
        right_model=PredModels.objects.get(encoding=model['encoding'],type=model['type'], method=model['method'],
                                           log=tr_log, prefix_length=prefix_length)
    except PredModels.DoesNotExist:
        split = model['split']
        if split['type'] == 'single':
            clust='noCluster'
        else:
            clust='Kmeans'
        config = {'key': 123,
                       'method': model['method'],
                       'encoding': model['encoding'],
                       'clustering': clust,
                       'prefix_length':prefix_length,
                       "rule": "remaining_time",
                       'threshold': 'default',
                       }
        try:
            split = Split.objects.get(type = 'single', original_log = tr_log)
        except Split.DoesNotExist:
            split = Split.objects.create(type = 'single', original_log = tr_log)
        j=Job.objects.create(config=config, split=split, type=model['type'])
        right_model = tr_calculate(j.to_dict(), redo=True)

    if job['type'] == CLASSIFICATION:
        results = classifier_run(run_df, right_model.to_dict())
    elif job['type'] == REGRESSION:
        results= regression_run(run_df, right_model.to_dict())
    elif job['type'] == NEXT_ACTIVITY:
        results = next_activity_run(run_df, right_model.to_dict())
    else:
        raise ValueError("Type not supported", job['type'])
    #print("End job {}, {} . Results {}".format(job['type'], get_run(job), results))
    return results

def getModelList(typed):
    models = PredModels.objects.all()        
    results = models.filter(type = typed)
    return results