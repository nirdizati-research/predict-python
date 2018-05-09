from hyperopt import Trials, STATUS_OK, tpe, fmin, STATUS_FAIL

from core.core import get_run, get_encoded_logs, run_by_type
from core.hyperopt_spaces import get_space

trial_nr = 0


def calculate_and_evaluate(args):
    global trial_nr
    if trial_nr % 20 == 0:
        print("Trial {}".format(trial_nr))
    trial_nr += 1
    local_job = global_job
    performance_metric = local_job['hyperopt']['performance_metric']
    method_conf_name = "{}.{}".format(local_job['type'], local_job['method'])
    local_job[method_conf_name] = {**local_job[method_conf_name], **args}
    multiplier = get_metric_multiplier(performance_metric)
    try:
        results, model_split = run_by_type(training_df, test_df, local_job)
        return {'loss': -results[performance_metric] * multiplier, 'status': STATUS_OK, 'results': results,
                'config': local_job[method_conf_name], 'model_split': model_split}
    except Exception:
        return {'loss': 100, 'status': STATUS_FAIL, 'results': {},
                'config': local_job[method_conf_name]}


def calculate_hyperopt(job):
    """ Main entry method for hyperopt calculations
        Returns the model for the best trial
    """
    print("Start hyperopt job {} with {}, performance_metric {}".format(job['type'], get_run(job),
                                                                        job['hyperopt']['performance_metric']))
    global training_df, test_df, global_job
    global_job = job
    training_df, test_df = get_encoded_logs(job)

    space = get_space(job)

    max_evals = job['hyperopt']['max_evals']
    trials = Trials()
    try:
        fmin(calculate_and_evaluate, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    except ValueError:
        raise ValueError("All jobs failed, cannot find best configuration")
    current_best = {'loss': 100, 'results': {}, 'config': {}}
    for t in trials:
        a = t['result']
        if current_best['loss'] > a['loss']:
            current_best = a

    print("End hyperopt job {}, {} . Results {}".format(job['type'], get_run(job), current_best['results']))
    return current_best['results'], current_best['config'], current_best['model_split']


def get_metric_multiplier(performance_metric):
    """Class methods have to be as good as possible. Some reg methods as bad as possible"""
    metric_map = {'rmse': -1, 'mae': -1, 'rscore': 1, 'acc': 1, 'f1score': 1, 'auc': 1, 'precsision': 1, 'recall': 1,
                  'true_positive': 1, 'true_negative': 1, 'false_positive': 1, 'false_negative': 1}
    return metric_map[performance_metric]
