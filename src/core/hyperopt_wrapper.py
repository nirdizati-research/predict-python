"""
hyperopt methods and functionalities
"""

from hyperopt import Trials, STATUS_OK, tpe, fmin, STATUS_FAIL

from src.core.core import get_encoded_logs, get_run, run_by_type
from src.core.hyperopt_spaces import _get_space

trial_number = 0


def calculate_hyperopt(job: dict) -> (dict, dict, dict):
    """main entry method for hyperopt calculations
    returns the predictive_model for the best trial

    :param job: job configuration
    :return: tuple containing the results, config and predictive_model split from the search

    """
    print("Start hyperopt job {} with {}, performance_metric {}".format(job['type'], get_run(job),
                                                                        job['hyperopt']['performance_metric']))
    global training_df, test_df, global_job
    global_job = job
    training_df, test_df = get_encoded_logs(job)

    space = _get_space(job)

    max_evals = job['hyperopt']['max_evals']
    trials = Trials()
    try:
        fmin(_calculate_and_evaluate, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    except ValueError:
        raise ValueError("All jobs failed, cannot find best configuration")
    current_best = {'loss': 100, 'results': {}, 'config': {}}
    for t in trials:
        a = t['result']
        if current_best['loss'] > a['loss']:
            current_best = a

    print("End hyperopt job {}, {} . Results {}".format(job['type'], get_run(job), current_best['results']))
    return current_best['results'], current_best['config'], current_best['model_split']


def get_metric_multiplier(performance_metric: int) -> int:
    """returns the multiplier to be used for each metric

    :param performance_metric: metric used (index)
    :return: metric multiplier associated
    """
    metric_map = {'rmse': -1, 'mae': -1, 'rscore': 1, 'acc': 1, 'f1score': 1, 'auc': 1, 'precision': 1, 'recall': 1,
                  'true_positive': 1, 'true_negative': 1, 'false_positive': 1, 'false_negative': 1, 'mape': -1}
    return metric_map[performance_metric]


def _calculate_and_evaluate(args) -> dict:
    global trial_number
    if trial_number % 20 == 0:
        print("Trial {}".format(trial_number))
    trial_number += 1
    local_job = global_job
    performance_metric = local_job['hyperopt']['performance_metric']
    method_conf_name = "{}.{}".format(local_job['type'], local_job['method'])
    local_job[method_conf_name] = {**local_job[method_conf_name], **args}
    multiplier = get_metric_multiplier(performance_metric)
    try:
        results, model_split = run_by_type(training_df.copy(), test_df.copy(), local_job)
        return {'loss': -results[performance_metric] * multiplier, 'status': STATUS_OK, 'results': results,
                'config': local_job[method_conf_name], 'model_split': model_split}
    except:
        return {'loss': 100, 'status': STATUS_FAIL, 'results': {},
                'config': local_job[method_conf_name]}
