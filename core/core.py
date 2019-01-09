import os
import pickle

from core.binary_classification import binary_classifier, binary_classifier_single_log
from core.constants import \
    CLASSIFICATION, REGRESSION, LABELLING, UPDATE
from core.multi_classification import multi_classifier, multi_classifier_single_log
from core.regression import regression, regression_single_log
from core.label_validation import label_task
from core.update_model import update_model
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

    train_set_fn = ('train_split-%d_pref-%d_encoding-%s_padding-%s_generation-%s.pickle' %
                         (job['split']['id'],
                          job['encoding'].prefix_length,
                          job['encoding'].method,
                          job['encoding'].padding,
                          job['encoding'].generation_type)
                         )
    test_set_fn = train_set_fn.replace('train', 'test')

    if os.path.isfile("labeled_log_cache/" + train_set_fn) and os.path.isfile("labeled_log_cache/" + test_set_fn):
        pickle_in = open("labeled_log_cache/" + train_set_fn, 'rb')
        training_df = pickle.load(pickle_in)

        pickle_in = open("labeled_log_cache/" + test_set_fn, 'rb')
        test_df = pickle.load(pickle_in)

    else:
        training_log, test_log, additional_columns = prepare_logs(job['split'])

        training_df, test_df = encode_label_logs(training_log, test_log, job['encoding'], job['type'], job['label'],
                                                 additional_columns=additional_columns)

        pickle_out = open("labeled_log_cache/" + train_set_fn, "wb")
        pickle.dump(training_df, pickle_out)
        pickle_out.close()

        pickle_out = open("labeled_log_cache/" + test_set_fn, "wb")
        pickle.dump(test_df, pickle_out)
        pickle_out.close()

    return training_df, test_df


def run_by_type(training_df, test_df, job):
    model_split = None
    if job['type'] == CLASSIFICATION:
        label_type = job['label'].type
        # Binary classification
        if label_type == REMAINING_TIME or label_type == ATTRIBUTE_NUMBER or label_type == DURATION:
            results, model_split = binary_classifier(training_df, test_df, job)
        elif label_type == NEXT_ACTIVITY or label_type == ATTRIBUTE_STRING:
            results, model_split = multi_classifier(training_df, test_df, job)
        else:
            raise ValueError("Label type not supported", label_type)
    elif job['type'] == REGRESSION:
        results, model_split = regression(training_df, test_df, job)
    elif job['type'] == LABELLING:
        results = label_task(training_df)
    elif job['type'] == UPDATE:
        results, model_split = update_model(training_df, test_df, job)
    else:
        raise ValueError("Type not supported", job['type'])

    print("End job {}, {} .".format(job['type'], get_run(job)))
    print("\tResults {} .".format(results))
    return results, model_split


def runtime_calculate(run_log, model):
    run_df = encode_label_log(run_log, model['encoding'], model['type'], model['label'])
    if model['type'] == CLASSIFICATION:
        label_type = model['label'].type
        if label_type == REMAINING_TIME or label_type == ATTRIBUTE_NUMBER or label_type == DURATION:
            results = binary_classifier_single_log(run_df, model)
        elif label_type == NEXT_ACTIVITY or label_type == ATTRIBUTE_STRING:
            results = multi_classifier_single_log(run_df, model)
        else:
            raise ValueError("Label type not supported", label_type)
    elif model['type'] == REGRESSION:
        results = regression_single_log(run_df, model)
    else:
        raise ValueError("Type not supported", model['type'])
    print("End job {}, {} . Results {}".format(model['type'], get_run(model), results))
    return results


def get_run(job):
    """Defines job identity"""
    if job['type'] == LABELLING:
        return job['encoding'].method + '_' + job['label'].type
    return job['method'] + '_' + job['encoding'].method + '_' + job['clustering'] + '_' + job['label'].type
