import logging
import time

import django_rq
from django_rq.decorators import job
from sklearn.externals import joblib

from src.clustering.models import ClusteringMethods
from src.core.core import calculate
from src.hyperparameter_optimization.hyperopt_wrapper import calculate_hyperopt
from src.hyperparameter_optimization.models import HyperparameterOptimizationMethods
from src.jobs.models import Job, JobStatuses, JobTypes, ModelType
from src.jobs.ws_publisher import publish

logger = logging.getLogger(__name__)


@job("default", timeout='1h')
def prediction_task(job_id):
    logger.info("Start prediction task ID {}".format(job_id))
    job = Job.objects.get(id=job_id)

    try:
        if (job.status == JobStatuses.CREATED.value and job.type != JobTypes.UPDATE.value) or \
            (job.status == JobStatuses.CREATED.value and job.type == JobTypes.UPDATE.value and
             job.incremental_train.status == JobStatuses.COMPLETED.value):

            job.status = JobStatuses.RUNNING.value
            job.save()
            start_time = time.time()
            if job.hyperparameter_optimizer is not None and \
                job.hyperparameter_optimizer.optimization_method != HyperparameterOptimizationMethods.NONE.value:
                result, model_split = hyperopt_task(job)
            else:
                result, model_split = calculate(job)
            elapsed_time = time.time() - start_time
            logger.info('\tJob took: {} in HH:MM:ss'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            if job.create_models:
                save_models(model_split, job)
            job.result = result
            job.status = JobStatuses.COMPLETED.value
        else:
            django_rq.enqueue(prediction_task, job.id)
    except Exception as e:
        logger.error(e)
        job.status = JobStatuses.ERROR.value
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()
        publish(job)


def save_models(models: dict, job: Job):
    logger.info("\tStart saving models of JOB {}".format(job.id))
    if job.clustering.clustering_method != ClusteringMethods.NO_CLUSTER.value:
        clusterer_filename = 'cache/model_cache/job_{}-split_{}-clusterer-{}-v0.sav'.format(
            job.id,
            job.split.id,
            job.type)
        joblib.dump(models[ModelType.CLUSTERER.value], clusterer_filename)
        job.clustering.model_path = clusterer_filename
        job.clustering.save()
        job.save()

    if job.type == JobTypes.UPDATE.value:
        job.type = JobTypes.PREDICTION.value  # TODO: Y am I doing this?
        predictive_model_filename = 'cache/model_cache/job_{}-split_{}-predictive_model-{}-v{}.sav'.format(
            job.id,
            job.split.id,
            job.type,
            str(time.time()))
    else:
        predictive_model_filename = 'cache/model_cache/job_{}-split_{}-predictive_model-{}-v0.sav'.format(
            job.id,
            job.split.id,
            job.type)
    joblib.dump(models[job.predictive_model.predictive_model], predictive_model_filename)
    job.predictive_model.model_path = predictive_model_filename
    job.predictive_model.save()
    job.save()


def hyperopt_task(job):
    # job_dict = job.to_dict()
    results, config, model_split = calculate_hyperopt(job)
    # method_conf_name = "{}.{}".format(job_dict['type'], job_dict['method'])
    # job.config[method_conf_name] = config
    return results, model_split
