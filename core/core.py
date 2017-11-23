from sklearn.model_selection import train_test_split

from core.classification import classifier
from core.constants import NEXT_ACTIVITY, \
    CLASSIFICATION, REGRESSION
from core.next_activity import next_activity
from core.regression import regression
from encoders.common import encode_logs
from logs.file_service import get_logs


def calculate(job):
    """ Main entry method for calculations"""
    results = None
    print("Start job {} with {}".format(job['type'], get_run(job)))
    log = get_logs(job['log'])[0]
    training_log, test_log = split_log(log)

    # Python dicts are bad
    if 'prefix_length' in job:
        prefix_length = job['prefix_length']
    else:
        prefix_length = 1

    training_df, test_df = encode_logs(training_log, test_log, job['encoding'], job['type'], prefix_length=prefix_length)

    if job['type'] == CLASSIFICATION:
        results = classifier(training_df, test_df, job)
    elif job['type'] == REGRESSION:
        results = regression(training_df, test_df, job)
    elif job['type'] == NEXT_ACTIVITY:
        results = next_activity(training_df, test_df, job)
    print("End job {}, {} . Results {}".format(job['type'], get_run(job), results))
    return results


def split_log(log: list, test_size=0.2, random_state=4):
    training_log, test_log = train_test_split(log, test_size=test_size, random_state=random_state)
    return training_log, test_log


def get_run(job):
    """Defines job identity"""
    if job['type'] == CLASSIFICATION:
        return run_classification(job['classification'], job['encoding'], job['clustering'])
    elif job['type'] == NEXT_ACTIVITY:
        return run_classification(job['classification'], job['encoding'], job['clustering'])
    elif job['type'] == REGRESSION:
        return run_regression(job['regression'], job['encoding'], job['clustering'])

def run_classification(classification, encoding, clustering):
    return classification + '_' + encoding + '_' + clustering


def run_regression(regression, encoding, clustering):
    return regression + '_' + encoding + '_' + clustering