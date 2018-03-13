from sklearn.model_selection import train_test_split

from core.classification import classifier
from core.constants import NEXT_ACTIVITY, \
    CLASSIFICATION, REGRESSION
from core.next_activity import next_activity
from core.regression import regression
from encoders.common import encode_logs
from logs.file_service import get_logs


def calculate(job, model):
    """ Main entry method for calculations"""
    print("Start job {} with {}".format(job['type'], get_run(job)))
    test_log = prepare_logs(job['split'])
    # Python dicts are bad
    
    test_df, prefix_length = encode_run_logs(training_log, test_log, job['encoding'], job['type'])

    if job['type'] == CLASSIFICATION:
        results = classifier(test_df, job, model.to_dict())
    elif job['type'] == REGRESSION:
        results = regression(test_df, job, model.to_dict())
    elif job['type'] == NEXT_ACTIVITY:
        results = next_activity(test_df, job, model.to_dict())
    else:
        raise ValueError("Type not supported", job['type'])
    print("End job {}, {} . Results {}".format(job['type'], get_run(job), results))
    return results


def prepare_logs(split: dict):
    """Returns test_log"""
    if split['type'] == 'single':
        path = split['original_log_path']
        test_log = get_logs(path)[0]
        test_log, _ = train_test_split(test_log, test_size=0)
    else:
        path = split['test_log_path']
        test_log = get_logs(path)[0]
        test_log, _ = train_test_split(test_log, test_size=0)
    return test_log


def split_log(log: list, test_size=0.2, random_state=4):
    training_log, test_log = train_test_split(log, test_size=test_size, random_state=random_state)
    #print(test_log[0][0])
    return training_log, test_log


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
