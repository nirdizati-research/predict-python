import json
import os
import time

from core.classification import classification, classification_single_log
from core.constants import CLASSIFICATION, REGRESSION, LABELLING
from core.constants import UPDATE
from core.regression import regression, regression_single_log
from core.update_model import update_model
from encoders.common import encode_label_logs, encode_label_log
from logs.splitting import prepare_logs
from utils.cache import load_from_cache, dump_to_cache, get_digested
from utils.file_service import save_result


def calculate(job):
    """ Main entry method for calculations"""
    print("Start job {} with {}".format(job['type'], get_run(job)))

    training_df, test_df = get_encoded_logs(job)
    results, model_split = run_by_type(training_df, test_df, job)
    return results, model_split


def get_encoded_logs(job: dict):
    processed_df_cache = ('split-%s_encoding-%s_type-%s_label-%s' % (json.dumps(job['split']),
                                                                     json.dumps(job['encoding']),
                                                                     json.dumps(job['type']),
                                                                     json.dumps(job['label'])))

    if os.path.isfile("labeled_log_cache/" + get_digested(processed_df_cache) + '.pickle'):

        print('Found Labeled Dataset in cache, loading...')
        training_df, test_df = load_from_cache(processed_df_cache, prefix="labeled_log_cache/")
        print('Done.')

    else:
        df_cache = ('split-%s' % (json.dumps(job['split'])))

        if os.path.isfile("labeled_log_cache/" + get_digested(df_cache) + '.pickle'):

            print('Found Dataset in cache, loading..')
            training_log, test_log, additional_columns = load_from_cache(df_cache, prefix="labeled_log_cache/")
            print('Dataset loaded.')

        else:
            training_log, test_log, additional_columns = prepare_logs(job['split'])

            dump_to_cache(df_cache, (training_log, test_log, additional_columns), prefix="labeled_log_cache/")

        training_df, test_df = encode_label_logs(training_log, test_log, job['encoding'], job['type'], job['label'],
                                                 additional_columns=additional_columns)

        dump_to_cache(processed_df_cache, (training_df, test_df), prefix="labeled_log_cache/")

    return training_df, test_df


def run_by_type(training_df, test_df, job):
    model_split = None

    start_time = time.time()
    if job['type'] == CLASSIFICATION:
        results, model_split = classification(training_df, test_df, job)
    elif job['type'] == REGRESSION:
        results, model_split = regression(training_df, test_df, job)
    elif job['type'] == LABELLING:
        results = _label_task(training_df)
    elif job['type'] == UPDATE:
        results, model_split = update_model(training_df, test_df, job)
    else:
        raise ValueError("Type not supported", job['type'])

    if job['type'] == CLASSIFICATION:
        save_result(results, job, start_time)

    print("End job {}, {} .".format(job['type'], get_run(job)))
    print("\tResults {} .".format(results))
    return results, model_split


def runtime_calculate(run_log, model):
    run_df = encode_label_log(run_log, model['encoding'], model['type'], model['label'])
    if model['type'] == CLASSIFICATION:
        results = classification_single_log(run_df, model)
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


def _label_task(df):
    """Calculates the distribution of labels in the data frame

    :return Dict of string and int {'label1': label1_count, 'label2': label2_count}
    """
    # Stupid but it works
    # True must be turned into 'true'
    json_value = df.label.value_counts().to_json()
    return json.loads(json_value)
