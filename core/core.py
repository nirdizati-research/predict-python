from sklearn.model_selection import train_test_split

from core.classification import classifier
from core.common import encode_log
from core.constants import NEXT_ACTIVITY, \
    CLASSIFICATION, REGRESSION
from core.next_activity import next_activity
from core.regression import regression
from encoders.common import encode_logs
from logs.file_service import get_logs


def calculate(job):
    """ Main entry method for calculations"""
    results = None
    print("Start job {} with {}".format(job.type, job.get_run()))
    log = get_logs(job.log)[0]
    training_log, test_log = split_log(log)
    training_df, test_df = encode_logs(training_log, test_log, job.encoding, job.type)
    df = encode_log(job.encoding, job.type, job.log)
    if job.type == CLASSIFICATION:
        results = classifier(training_df, test_df, job)
    elif job.type == REGRESSION:
        results = regression(training_df, test_df, job)
    elif job.type == NEXT_ACTIVITY:
        results = next_activity(df, job)
    print("End job {}, {} . Results {}".format(job.type, job.get_run(), results))
    return results


def split_log(log: list, test_size=0.2, random_state=4):
    training_log, test_log = train_test_split(log, test_size=test_size, random_state=random_state)
    return training_log, test_log
