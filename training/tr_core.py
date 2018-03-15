from sklearn.model_selection import train_test_split
from django.core.files import File

from training.tr_classification import tr_classifier
from core.constants import NEXT_ACTIVITY, \
    CLASSIFICATION, REGRESSION
from training.tr_next_activity import tr_next_activity
from training.tr_regression import tr_regression
from encoders.common import encode_training_logs, encode_logs, encode_one_training_logs
from logs.models import Log
from logs.file_service import get_logs
from logs.views import save_file
from sklearn.externals import joblib
from training.models import PredModels, Split


def calculate(job, redo=False):
    """ Main entry method for calculations"""
    print("Start job {} with {}".format(job['type'], get_run(job)))
    training_log, path = prepare_logs(job['split'])
    log = Log.objects.get(path = path)

        # Python dicts are bad
    
    if redo:
        train_df = encode_one_training_logs(training_log, job['encoding'], job['type'], job['prefix_length'])
        models = work(log, job, job['prefix_length'], train_df)
    else:
        training_df, prefix_length = encode_training_logs(training_log, job['encoding'], job['type'])
        for i, train_df in training_df.items():
            models = work(log, job, i, train_df)
    return models


def prepare_logs(split: dict):
    """Returns training_log"""
    if split['type'] == 'single':
        path = split['original_log_path']
        training_log = get_logs(path)[0]
        training_log, _ =train_test_split(training_log, test_size=0)
    else:
        path = split['training_log_path']
        training_log = get_logs(path)[0]
        training_log, _ =train_test_split(training_log, test_size=0)
    return training_log, path

def work(log, job, i, train_df):
    if job['type'] == CLASSIFICATION:
        split = tr_classifier(train_df, job)
    elif job['type'] == REGRESSION:
        split = tr_regression(train_df, job)
    elif job['type'] == NEXT_ACTIVITY:
        split = tr_next_activity(train_df, job)
    else:
        raise ValueError("Type not supported", job['type'])
    print("End job {}, {}".format(job['type'], get_run(job)))

    if split['type'] =='single':
        filename_model = 'model_cache/{}-model-{}.sav'.format(log.name,i)
        joblib.dump(split['model'], filename_model)
        try:
            split=Split.objects.get(type=split['type'], model_path = filename_model)
        except Split.DoesNotExist:
            split = Split.objects.create(type = split['type'], model_path = filename_model)
        try:
            models = PredModels.objects.get(split=split, type=job['type'], log = log, prefix_length = i, encoding = job['encoding'],
                                       method = job['method'])
        except:
            models = PredModels.objects.create(split = split, type=job['type'], log = log, prefix_length = i, encoding = job['encoding'],
                                       method = job['method'])
    elif split['type'] == 'double':
        filename_model = 'model_cache/{}-model-{}.sav'.format(log.name, i)
        filename_estimator = 'model_cache/{}-estimator.sav'.format(log.name)
        joblib.dump(split['model'], filename_model)
        joblib.dump(split['estimator'], filename_estimator)
        try:
            split=Split.objects.get(type=split['type'], model_path = filename_model, kmean_path=filename_estimator)
        except Split.DoesNotExist:
            split = Split.objects.create(type = split['type'], model_path = filename_model, kmean_path = filename_estimator)
        
    try:
        models = PredModels.objects.get(split=split, type=job['type'], log = log, prefix_length = i, encoding = job['encoding'],
                                       method = job['method'])
    except PredModels.DoesNotExist:
        models = PredModels.objects.create(split = split, type=job['type'], log = log, prefix_length = i, encoding = job['encoding'],
                                       method = job['method'])
    return models


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