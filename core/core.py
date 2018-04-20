from core.classification import classifier
from core.constants import NEXT_ACTIVITY, \
    CLASSIFICATION, REGRESSION, ZERO_PADDING
from core.next_activity import next_activity
from core.regression import regression
from encoders.common import encode_logs
from logs.splitting import prepare_logs


def calculate(job):
    """ Main entry method for calculations"""
    print("Start job {} with {}".format(job['type'], get_run(job)))
    training_log, test_log = prepare_logs(job['split'])

    # Python dicts are bad
    if 'prefix_length' in job:
        prefix_length = job['prefix_length']
    else:
        prefix_length = 1
    zero_padding = True if job['padding'] is ZERO_PADDING else False
    add_elapsed_time = job.get('add_elapsed_time', True)

    training_df, test_df = encode_logs(training_log, test_log, job['encoding'], job['type'],
                                       prefix_length=prefix_length, zero_padding=zero_padding,
                                       add_elapsed_time=add_elapsed_time)

    if job['type'] == CLASSIFICATION:
        results, model_split = classifier(training_df, test_df, job)
    elif job['type'] == REGRESSION:
        results, model_split = regression(training_df, test_df, job)
    elif job['type'] == NEXT_ACTIVITY:
        results, model_split = next_activity(training_df, test_df, job)
    else:
        raise ValueError("Type not supported", job['type'])
    print("End job {}, {} . Results {}".format(job['type'], get_run(job), results))
    return results, model_split


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
