import numpy as np
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll.base import scope

from core.core import get_run, get_encoded_logs, run_by_type

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
    results, _ = run_by_type(training_df, test_df, local_job)

    return {'loss': -results[performance_metric], 'status': STATUS_OK, 'results': results,
            'config': local_job[method_conf_name]}


def calculate_hyperopt(job):
    """ Main entry method for hyperopt calculations"""
    print("Start hyperopt job {} with {}, performance_metric {}".format(job['type'], get_run(job),
                                                                        job['hyperopt']['performance_metric']))
    global training_df, test_df, global_job
    global_job = job
    training_df, test_df = get_encoded_logs(job)

    space = {'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
             'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
             'max_features': hp.uniform('max_features', 0, 1)}

    max_evals = job['hyperopt']['max_evals']
    trials = Trials()
    fmin(calculate_and_evaluate, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    current_best = {'loss': 100}
    for t in trials:
        a = t['result']
        if current_best['loss'] > a['loss']:
            current_best = a

    print("End hyperopt job {}, {} . Results {}".format(job['type'], get_run(job), current_best['results']))
    return current_best['results'], current_best['config']
