from sklearn.model_selection import train_test_split
from django.core.files import File

from training.tr_classification import tr_classifier
from core.constants import NEXT_ACTIVITY, \
    CLASSIFICATION, REGRESSION
from training.tr_next_activity import tr_next_activity
from training.tr_regression import tr_regression
from encoders.common import encode_training_logs, encode_logs
from logs.models import Log
from logs.file_service import get_logs
from logs.views import save_file
from sklearn.externals import joblib
from training.models import PredModels, Split


def calculate(job):
    """ Main entry method for calculations"""
    print("Start job {} with {}".format(job['type'], get_run(job)))
    training_log, path = prepare_logs(job['split'])
    log = Log.objects.get(path = path)

        # Python dicts are bad
    if 'prefix_length' in job:
        prefix_length = job['prefix_length']
    else:
        prefix_length = 1
    
    training_log, _ =train_test_split(training_log, test_size=0)
    
    training_df = encode_training_logs(training_log, job['encoding'], job['type'],
                                       prefix_length=prefix_length)
    
    if job['type'] == CLASSIFICATION:
        split = tr_classifier(training_df, job)
    elif job['type'] == REGRESSION:
        split = tr_regression(training_df, job)
    elif job['type'] == NEXT_ACTIVITY:
        split = tr_next_activity(training_df, job)
    else:
        raise ValueError("Type not supported", job['type'])
    print("End job {}, {}".format(job['type'], get_run(job)))

    if split['type'] =='single':
        filename_model = 'model_cache/{}-model.sav'.format(log.name)
        joblib.dump(split['model'], filename_model)
        split = Split.objects.create(type = split['type'], model_path = filename_model)
        models = PredModels.objects.create(split = split, type=job['type'], log = log, prefix_length = prefix_length, encoding = job['encoding'],
                                       method = job['method'])
    elif split['type'] == 'double':
        filename_model = 'model_cache/{}-model.sav'.format(log.name)
        filename_estimator = 'model_cache/{}-estimator.sav'.format(log.name)
        joblib.dump(split['model'], filename_model)
        joblib.dump(split['estimator'], filename_estimator)
        split = Split.objects.create(type = split['type'], model_path = filename_model, kmean_path = filename_estimator)
        models = PredModels.objects.create(split = split, type=job['type'], log = log, prefix_length = prefix_length, encoding = job['encoding'],
                                       method = job['method'])
    return models


def prepare_logs(split: dict):
    """Returns training_log and test_log"""
    if split['type'] == 'single':
        path = split['original_log_path']
        training_log = get_logs(path)[0]
    return training_log, path


def get_run(job):
    """Defines job identity"""
    if job['type'] == CLASSIFICATION:
        return run_identity(job['method'], job['encoding'], job['clustering'])
    elif job['type'] == NEXT_ACTIVITY:
        return run_identity(job['method'], job['encoding'], job['clustering'])
    elif job['type'] == REGRESSION:
        return run_identity(job['method'], job['encoding'], job['clustering'])


def run_identity(method, encoding, clustering):
    return method + '_' + encoding + '_' + clustering