import csv
import os
import pickle
import time

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

    BALANCED = False

    train_set_fn = 'train_split-' + str(job['split']['id']) + '.pickle'
    test_set_fn = train_set_fn.replace('train', 'test')
    additional_columns_fn = test_set_fn.replace('test', 'additional_columns')

    if os.path.isfile("labeled_log_cache/" + train_set_fn) and \
        os.path.isfile("labeled_log_cache/" + test_set_fn) and \
        os.path.isfile("labeled_log_cache/" + additional_columns_fn):

        print('Found Dataset in cache, loading..')
        pickle_in = open("labeled_log_cache/" + train_set_fn, 'rb')
        training_log = pickle.load(pickle_in)

        pickle_in = open("labeled_log_cache/" + test_set_fn, 'rb')
        test_log = pickle.load(pickle_in)

        pickle_in = open("labeled_log_cache/" + additional_columns_fn, 'rb')
        additional_columns = pickle.load(pickle_in)
        print('Dataset loaded.')

    else:
        training_log, test_log, additional_columns = prepare_logs(job['split'])

        pickle_out = open("labeled_log_cache/" + train_set_fn, "wb")
        pickle.dump(training_log, pickle_out)
        pickle_out.close()

        pickle_out = open("labeled_log_cache/" + test_set_fn, "wb")
        pickle.dump(test_log, pickle_out)
        pickle_out.close()

        pickle_out = open("labeled_log_cache/" + additional_columns_fn, "wb")
        pickle.dump(additional_columns, pickle_out)
        pickle_out.close()

    training_df, test_df = encode_label_logs(training_log, test_log, job['encoding'], job['type'], job['label'],
                                             additional_columns=additional_columns, balance=BALANCED)

    return training_df, test_df


def run_by_type(training_df, test_df, job):
    model_split = None

    start_time = time.time()

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

    result = [
        results['f1score'],
        results['acc'],
        results['true_positive'],
        results['true_negative'],
        results['false_negative'],
        results['false_positive'],
        results['precision'],
        results['recall'],
        results['auc']
    ]
    result += [job['encoding'][index] for index in range(len(job['encoding']))]
    result += [job['label'][index] for index in range(len(job['label']))]
    result += [job['incremental_train'][index] for index in job['incremental_train'].keys()]
    result += [job['hyperopt'][index] for index in job['hyperopt'].keys()]
    result += [job['clustering']]
    result += [job['split'][index] for index in job['split'].keys()]
    result += [job['type']]
    result += [job[job['type'] + '.' + job['method']][index] for index in job[job['type'] + '.' + job['method']].keys()]
    result += [str(time.time() - start_time)]

    with open('results/' + job['type'] + '-' + job['method'] + '_result.csv', 'a+') as log_result_file:
        writer = csv.writer(log_result_file)
        if sum(1 for line in open('results/' + job['type'] + '-' + job['method'] + '_result.csv')) == 0:
            writer.writerow(['f1score',
                             'acc',
                             'true_positive',
                             'true_negative',
                             'false_negative',
                             'false_positive',
                             'precision',
                             'recall',
                             'auc'] +
                            list(job['encoding']._fields) +
                            list(job['label']._fields) +
                            list(job['incremental_train'].keys()) +
                            list(job['hyperopt'].keys()) +
                            ['clustering'] +
                            list(job['split'].keys()) +
                            ['type'] +
                            list(job[job['type'] + '.' + job['method']].keys()) +
                            ['time_elapsed(s)']
                            )
        writer.writerow(result)

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
