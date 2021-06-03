import logging
import time

import django_rq
from django_rq.decorators import job
from sklearn.externals import joblib

from src.clustering.models import ClusteringMethods
from src.core.core import calculate
from src.hyperparameter_optimization.hyperopt_wrapper import calculate_hyperopt
from src.hyperparameter_optimization.models import HyperparameterOptimizationMethods
from src.jobs.job_creator import set_model_name
from src.jobs.models import Job, JobStatuses, JobTypes, ModelType
from src.jobs.ws_publisher import publish

logger = logging.getLogger(__name__)


@job("default", timeout='100h')
def prediction_task(job_id, do_publish_result=True):
    """Predict the task using the given job id

    :param job_id:
    :param do_publish_result:
    """

    logger.info("Start prediction task ID {}".format(job_id))
    job = Job.objects.get(id=job_id)

    try:
        if (job.status == JobStatuses.CREATED.value and job.type != JobTypes.UPDATE.value) or \
           (job.status == JobStatuses.CREATED.value and job.type == JobTypes.UPDATE.value and
            job.incremental_train.status == JobStatuses.COMPLETED.value):

            job.status = JobStatuses.RUNNING.value
            job.save()
            job_start_time = time.time()
            if job.hyperparameter_optimizer is not None and \
                job.hyperparameter_optimizer.optimization_method != HyperparameterOptimizationMethods.NONE.value:
                result, _, model_split = calculate_hyperopt(job)
            else:
                result, model_split = calculate(job)
            job_elapsed_time = time.time() - job_start_time
            logger.info('\tJob took: {} in HH:MM:ss'.format(time.strftime("%H:%M:%S", time.gmtime(job_elapsed_time))))
            if job.create_models:
                save_models(model_split, job)
            job.result = result
            # Evaluation.init(
            #     job.predictive_model.predictive_model,
            #     results
            # )
            job.status = JobStatuses.COMPLETED.value
        elif job.status in [JobStatuses.COMPLETED.value, JobStatuses.ERROR.value, JobStatuses.RUNNING.value]:
            django_rq.enqueue(prediction_task, job.id)
    except Exception as e:
        logger.error(e)
        job.status = JobStatuses.ERROR.value
        job.error = str(e.__repr__())
        raise e
    finally:
        job.save()
        if do_publish_result:
            publish(job)


def save_models(models: dict, job: Job):
    """Saves the given models in the given job configuration

    :param models:
    :param job:
    """
    set_model_name(job)

    logger.info("\tStart saving models of JOB {}".format(job.id))
    if job.clustering.clustering_method != ClusteringMethods.NO_CLUSTER.value:
        joblib.dump(models[ModelType.CLUSTERER.value], job.clustering.model_path)
        job.clustering.save()
        job.save()

    joblib.dump(models[job.predictive_model.predictive_model], job.predictive_model.model_path)
    job.predictive_model.save()
    job.save()

