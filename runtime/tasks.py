from django_rq.decorators import job

from predModels.models import PredModels
from logs.models import Log, Split
from logs.file_service import get_logs
from core.classification import classifier_single_log
from core.next_activity import next_activity_single_log
from core.regression import regression_single_log
from encoders.common import encode_log
from jobs.models import Job, CREATED, RUNNING, COMPLETED, ERROR
from core.constants import CLASSIFICATION, NEXT_ACTIVITY, REGRESSION
from core.core import get_run

@job("default", timeout='1h')
def runtime_task(job, model):
    print("Start runtime task ID {}".format(job.pk))
    try:
        if job.status == CREATED:
            job.status = RUNNING
            job.save()
            log = Log.objects.get(pk=job['log_id'])
            run_log = get_logs(log.path)[0]
            result = calculate(run_log,model.to_dict())
            job.result = result
            job.status = COMPLETED
    except Exception as e:
        print("error " + str(e.__repr__()))
        job.status = ERROR
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()
        
def calculate(run_log,model):
    """ Main entry method for calculations"""
    
    # Python dicts are bad
    if 'prefix_length' in model:
        prefix_length = model['prefix_length']
    else:
        prefix_length = 1
    zero_padding = True if model['padding'] is ZERO_PADDING else False
    
    run_df= encode_log(run_log, model['config']['encoding'], model['type'], 
                       add_label=False, zero_padding=zero_padding)

    if job['type'] == CLASSIFICATION:
        results = classifier_single_log(run_df, model)
    elif job['type'] == REGRESSION:
        results= regression_single_log(run_df, model)
    elif job['type'] == NEXT_ACTIVITY:
        results = next_activity_single_log(run_df, model)
    else:
        raise ValueError("Type not supported", job['type'])
    print("End job {}, {} . Results {}".format(job['type'], get_run(job), results))
    return results
