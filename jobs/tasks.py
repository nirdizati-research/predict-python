import time

from django_rq.decorators import job
from sklearn.externals import joblib

from core.constants import KMEANS, UPDATE, CLASSIFICATION
from core.core import calculate
from core.hyperopt_wrapper import calculate_hyperopt
from jobs.models import Job, CREATED, RUNNING, COMPLETED, ERROR
from jobs.ws_publisher import publish
from predModels.models import ModelSplit, PredModels


@job("default", timeout='1h')
def prediction_task(job_id):
    print("Start prediction task ID {}".format(job_id))
    job = Job.objects.get(id=job_id)
    try:
        if job.status == CREATED:
            job.status = RUNNING
            job.save()
            start_time = time.time()
            if job.config.get('hyperopt', {}).get('use_hyperopt', False):
                result, model_split = hyperopt_task(job)
            else:
                result, model_split = calculate(job.to_dict())
            elapsed_time = time.time() - start_time
            print('\tJob took: {} in HH:MM:ss'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            if job.config.get('create_models', False):
                save_models(model_split, job)
            job.result = result
            job.status = COMPLETED
    except Exception as e:
        print("error " + str(e.__repr__()))
        job.status = ERROR
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()
        publish(job)


def save_models(to_model_split, job):
    print("Start saving models of JOB {}".format(job.id))
    jobsplit = job.split
    if jobsplit.type == 'single':
        log = jobsplit.original_log
    else:
        log = jobsplit.training_log
    if job.type == UPDATE:
        job.type = CLASSIFICATION
        filename_model = 'model_cache/job_{}-split_{}-model-{}-v{}.sav'.format(job.id, job.split.id, job.type,
                                                                               str(to_model_split['versioning'] + 1))
    else:
        filename_model = 'model_cache/job_{}-split_{}-model-{}-v0.sav'.format(job.id, job.split.id, job.type)
    joblib.dump(to_model_split['model'], filename_model)
    model_split, created = ModelSplit.objects.get_or_create(type=to_model_split['type'], model_path=filename_model,
                                                            predtype=job.type)
    if to_model_split['type'] == KMEANS:
        filename_estimator = 'model_cache/job_{}-split_{}-estimator-{}.sav'.format(job.id, job.split.id, job.type)
        joblib.dump(to_model_split['estimator'], filename_estimator)
        model_split.estimator_path = filename_estimator
        model_split.save()
    PredModels.objects.create(pk=job.id, split=model_split, type=job.type, log=log, config=job.config)


def hyperopt_task(job):
    job_dict = job.to_dict()
    results, config, model_split = calculate_hyperopt(job_dict)
    method_conf_name = "{}.{}".format(job_dict['type'], job_dict['method'])
    job.config[method_conf_name] = config
    return results, model_split
