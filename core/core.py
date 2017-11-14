from core.classification import classifier
from core.common import encode_log
from core.constants import NEXT_ACTIVITY, \
    CLASSIFICATION, REGRESSION
from core.next_activity import next_activity
from core.regression import regression


def calculate(job):
    """ Main entry method for calculations"""
    results = None
    print("Start job {} with {}".format(job.type, job.get_run()))
    df = encode_log(job.encoding, job.type)
    if job.type == CLASSIFICATION:
        results = classifier(df, job)
    elif job.type == REGRESSION:
        results = regression(df, job)
    elif job.type == NEXT_ACTIVITY:
        results = next_activity(df, job)
    print("End job {} with {}".format(job.type, job.get_run()))
    print(results)
    return results
