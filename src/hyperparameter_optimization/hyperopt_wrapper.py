"""
hyperopt methods and functionalities
"""
import logging
import time
from datetime import timedelta

import hyperopt
from hyperopt import Trials, STATUS_OK, fmin, STATUS_FAIL

from src.core.core import get_encoded_logs, get_run, run_by_type
from src.evaluation.models import Evaluation
from src.hyperparameter_optimization.hyperopt_spaces import _get_space
from src.hyperparameter_optimization.models import HyperOptAlgorithms, HyperOptLosses
from src.jobs.models import Job
from src.predictive_model.classification.classification import _test as classification_test,\
    _check_is_binary_classifier, _prepare_results as classification_prepare_results
from src.predictive_model.regression.regression import _test as regression_test
from src.predictive_model.models import PredictiveModel, PredictiveModels
from src.utils.django_orm import duplicate_orm_row
from src.utils.result_metrics import _prepare_results as regression_prepare_results

logger = logging.getLogger(__name__)

trial_number = 0

holdout = False #TODO evaluate on validation set

OPTIMISATION_ALGORITHM = {
    HyperOptAlgorithms.RANDOM_SEARCH.value: hyperopt.rand,
    HyperOptAlgorithms.TPE.value: hyperopt.tpe,
    HyperOptAlgorithms.ADAPTIVE_TPE.value: hyperopt.atpe,
}


def _retrieve_train_validate_test(local_train_df, local_test_df):
    validation_df = local_test_df
    # test_df = training_df.sample(frac=.2)
    local_test_df = local_test_df.tail(int(len(local_train_df) * 20 / 100))
    local_train_df = local_train_df.drop(local_test_df.index)
    return local_train_df, validation_df, local_test_df


def _run_hyperoptimisation(space, algorithm_suggest, max_evaluations, trials):
    try:
        return fmin(_calculate_and_evaluate, space, algo=algorithm_suggest, max_evals=max_evaluations, trials=trials)
    except ValueError:
        raise ValueError("All jobs failed, cannot find best configuration")


def _test_best_candidate(current_best, job_labelling_type, job_type):
    if job_type == PredictiveModels.CLASSIFICATION.value:
        return classification_test(current_best['model_split'], validation_df.drop(['trace_id'], 1),
                     evaluation=True, is_binary_classifier=_check_is_binary_classifier(job_labelling_type))
    elif job_type == PredictiveModels.REGRESSION.value:
        return regression_test(current_best['model_split'], validation_df.drop(['trace_id'], 1)), 0


def run_hyperopt(job, original_training_df, original_test_df):
    global train_df, test_df, global_job, validation_df
    global_job = job
    train_df, test_df = original_training_df.copy(), original_test_df.copy()

    train_df, validation_df, test_df = _retrieve_train_validate_test(train_df, test_df)

    space = _get_space(job)

    max_evaluations = 1000
    trials = Trials()

    algorithm = OPTIMISATION_ALGORITHM[
        job.hyperparameter_optimizer.__getattribute__(
            job.hyperparameter_optimizer.optimization_method.lower()
        ).algorithm_type
    ]

    _run_hyperoptimisation(space, algorithm.suggest, max_evaluations, trials)

    best_candidate = trials.best_trial['result']

    results_df, auc = _test_best_candidate(best_candidate, job.labelling.type, job.predictive_model.predictive_model)
    if job.predictive_model.predictive_model == PredictiveModels.CLASSIFICATION.value:
        return classification_prepare_results(results_df, auc)
    else:
        return regression_prepare_results(results_df, job.labelling)


def calculate_hyperopt(job: Job) -> (dict, dict, dict):
    """main entry method for hyperopt calculations
    returns the predictive_model for the best trial

    :param job: job configuration
    :return: tuple containing the results, config and predictive_model split from the search
    """

    logger.info("Start hyperopt job {} with {}, performance_metric {}".format(
        job.type, get_run(job),
        job.hyperparameter_optimizer.__getattribute__(
            job.hyperparameter_optimizer.optimization_method.lower()
        ).performance_metric) #Todo: WHY DO I NEED TO GET HYPEROPT?
    )

    global train_df, test_df, global_job, validation_df
    global_job = job
    train_df, test_df = get_encoded_logs(job)
    train_df, validation_df, test_df = _retrieve_train_validate_test(train_df, test_df)

    train_start_time = time.time()

    space = _get_space(job)

    max_evaluations = job.hyperparameter_optimizer.__getattribute__(
            job.hyperparameter_optimizer.optimization_method.lower()
        ).max_evaluations #Todo: WHY DO I NEED TO GET HYPEROPT?
    trials = Trials()

    algorithm = algorithm = OPTIMISATION_ALGORITHM[
        job.hyperparameter_optimizer.__getattribute__(
            job.hyperparameter_optimizer.optimization_method.lower()
        ).algorithm_type
    ]
    _run_hyperoptimisation(space, algorithm.suggest, max_evaluations, trials)

    best_candidate = trials.best_trial['result']

    job.predictive_model = PredictiveModel.objects.filter(pk=best_candidate['predictive_model_id'])[0]
    job.predictive_model.save()
    job.save()

    best_candidate['results']['elapsed_time'] = timedelta(seconds=time.time() - train_start_time)  # todo find better place for this
    job.evaluation.elapsed_time = best_candidate['results']['elapsed_time']
    job.evaluation.save()

    results_df, auc = _test_best_candidate(best_candidate, job.labelling.type, job.predictive_model.predictive_model)
    if job.predictive_model.predictive_model == PredictiveModels.CLASSIFICATION.value:
        results = classification_prepare_results(results_df, auc)
    else:
        results = regression_prepare_results(results_df, job.labelling)
    results['elapsed_time'] = job.evaluation.elapsed_time
    job.evaluation = Evaluation.init(
        job.predictive_model.predictive_model,
        results,
        len(set(validation_df['label'])) <= 2
    )
    job.evaluation.save()
    job.save()

    logger.info("End hyperopt job {}, {}. \n\tResults on test {}. \n\tResults on validation {}.".format(job.type, get_run(job), best_candidate['results'], results))
    return results, best_candidate['config'], best_candidate['model_split']


def _get_metric_multiplier(performance_metric: int) -> int:
    """returns the multiplier to be used for each metric

    :param performance_metric: metric used
    :return: metric multiplier associated
    """
    metric_map = {HyperOptLosses.RMSE.value: -1,
                  HyperOptLosses.MAE.value: -1,
                  HyperOptLosses.RSCORE.value: 1,
                  HyperOptLosses.ACC.value: 1,
                  HyperOptLosses.F1SCORE.value: 1,
                  HyperOptLosses.AUC.value: 1,
                  HyperOptLosses.PRECISION.value: 1,
                  HyperOptLosses.RECALL.value: 1,
                  HyperOptLosses.TRUE_POSITIVE.value: 1,
                  HyperOptLosses.TRUE_NEGATIVE.value: 1,
                  HyperOptLosses.FALSE_POSITIVE.value: 1,
                  HyperOptLosses.FALSE_NEGATIVE.value: 1,
                  HyperOptLosses.MAPE.value: -1}
    return metric_map[performance_metric]


def _calculate_and_evaluate(args) -> dict:
    global trial_number
    if trial_number % 20 == 0:
        logger.info("Trial {}".format(trial_number))
    trial_number += 1
    local_job = global_job

    predictive_model = local_job.predictive_model.predictive_model
    prediction_method = local_job.predictive_model.prediction_method

    model_config = {'predictive_model': predictive_model, 'prediction_method': prediction_method, **args}

    new_predictive_model = PredictiveModel.init(model_config)
    local_job.predictive_model = duplicate_orm_row(new_predictive_model)
    local_job.predictive_model.save()
    local_job.save()
    # local_job = duplicate_orm_row(local_job) #TODO not sure it is ok to have this here.

    performance_metric = local_job.hyperparameter_optimizer.__getattribute__(
        local_job.hyperparameter_optimizer.optimization_method.lower()
    ).performance_metric
    multiplier = _get_metric_multiplier(performance_metric)

    current_training_df, current_validation_df = train_df.copy(), validation_df.copy()

    try:
        results, model_split = run_by_type(current_training_df, current_validation_df, local_job)
        del current_training_df
        del current_validation_df
        return {
            'loss': -results[performance_metric] * multiplier,
            'status': STATUS_OK,
            'results': results,
            'predictive_model_id': local_job.predictive_model.pk,
            'model_split': model_split,
            'config': model_config}
    except Exception as e:
        logger.error(e)
        return {
            'loss': 100,
            'status': STATUS_FAIL,
            'results': {},
            'predictive_model_id': {},
            'model_split': {},
            'config': {}}
