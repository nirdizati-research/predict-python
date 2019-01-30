from core.classification import classification, classification_single_log
from core.constants import CLASSIFICATION, REGRESSION, LABELLING
from core.label_validation import label_task
from core.regression import regression, regression_single_log
from encoders.common import encode_label_logs, REMAINING_TIME, ATTRIBUTE_NUMBER, ATTRIBUTE_STRING, NEXT_ACTIVITY, \
    encode_label_log, DURATION
from logs.splitting import prepare_logs


def calculate(job):
    """ Main entry method for calculations"""
    print("Start job {} with {}".format(job['type'], get_run(job)))

    training_df, test_df = get_encoded_logs(job)
    results, model_split = run_by_type(training_df, test_df, job)
    return results, model_split


def get_encoded_logs(job: dict):
    training_log, test_log, additional_columns = prepare_logs(job['split'])

    training_df, test_df = encode_label_logs(training_log, test_log, job['encoding'], job['type'], job['label'],
                                             additional_columns=additional_columns)
    return training_df, test_df


def run_by_type(training_df, test_df, job):
    model_split = None
    label_type = job['label'].type

    if job['type'] == CLASSIFICATION:
        is_binary_classifier = check_is_binary_classifier(label_type)
        results, model_split = classification(training_df, test_df, job, is_binary_classifier)
    elif job['type'] == REGRESSION:
        results, model_split = regression(training_df, test_df, job)
    elif job['type'] == LABELLING:
        results = label_task(training_df)
    else:
        raise ValueError("Type not supported", job['type'])
    print("End job {}, {} . Results {}".format(job['type'], get_run(job), results))
    return results, model_split


def runtime_calculate(run_log, model):
    run_df = encode_label_log(run_log, model['encoding'], model['type'], model['label'])
    if model['type'] == CLASSIFICATION:
        label_type = model['label'].type
        is_binary_classifier = check_is_binary_classifier(label_type)
        results = classification_single_log(run_df, model, is_binary_classifier)
    elif model['type'] == REGRESSION:
        results = regression_single_log(run_df, model)
    else:
        raise ValueError("Type not supported", model['type'])
    print("End job {}, {} . Results {}".format(model['type'], get_run(model), results))
    return results


def check_is_binary_classifier(label_type):
    if label_type in [REMAINING_TIME, ATTRIBUTE_NUMBER, DURATION]:
        return True
    elif label_type in [NEXT_ACTIVITY, ATTRIBUTE_STRING]:
        return False
    else:
        raise ValueError("Label type not supported", label_type)


def get_run(job):
    """Defines job identity"""
    if job['type'] == LABELLING:
        return job['encoding'].method + '_' + job['label'].type
    return job['method'] + '_' + job['encoding'].method + '_' + job['clustering'] + '_' + job['label'].type
