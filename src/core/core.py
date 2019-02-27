import json
import os
import time

from pandas import DataFrame

from src.core.constants import TIME_SERIES_PREDICTION, REGRESSION, CLASSIFICATION, LABELLING, UPDATE
from src.encoding.common import encode_label_log, encode_label_logs
from src.predictive_model.classification.classification import classification_single_log, update_and_test, \
    classification
from src.predictive_model.regression.regression import regression, regression_single_log
from src.predictive_model.time_series_prediction.time_series_prediction import time_series_prediction_single_log, \
    time_series_prediction
from src.split.splitting import prepare_logs
from src.utils.cache import load_from_cache, dump_to_cache, get_digested
from src.utils.file_service import save_result


def calculate(job: dict) -> (dict, dict):
    """main entry point for calculations

    encodes the logs based on the given configuration and runs the selected task
    :param job: job configuration
    :return: results and predictive_model split

    """
    print("Start job {} with {}".format(job['type'], get_run(job)))
    training_df, test_df = get_encoded_logs(job)
    results, model_split = run_by_type(training_df, test_df, job)
    return results, model_split


def get_encoded_logs(job: dict, use_cache: bool = True) -> (DataFrame, DataFrame):
    """returns the encoded logs

    returns the training and test DataFrames encoded using the given job configuration, loading from cache if possible
    :param job: job configuration
    :param use_cache: load or not saved datasets from cache
    :return: training and testing DataFrame

    """
    if use_cache:
        processed_df_cache = ('split-%s_encoding-%s_type-%s_label-%s' % (json.dumps(job['split']),
                                                                         json.dumps(job['encoding']),
                                                                         json.dumps(job['type']),
                                                                         json.dumps(job['label'])))

        if os.path.isfile("cache/labeled_log_cache/" + get_digested(processed_df_cache) + '.pickle'):

            print('Found Labeled Dataset in cache, loading...')
            training_df, test_df = load_from_cache(processed_df_cache, prefix="cache/labeled_log_cache/")
            print('Done.')

        else:
            df_cache = ('split-%s' % (json.dumps(job['split'])))

            if os.path.isfile("cache/labeled_log_cache/" + get_digested(df_cache) + '.pickle'):

                print('Found Dataset in cache, loading..')
                training_log, test_log, additional_columns = load_from_cache(df_cache,
                                                                             prefix="cache/labeled_log_cache/")
                print('Dataset loaded.')

            else:
                training_log, test_log, additional_columns = prepare_logs(job['split'])
                dump_to_cache(df_cache, (training_log, test_log, additional_columns), prefix="cache/labeled_log_cache/")

            training_df, test_df = encode_label_logs(training_log, test_log, job['encoding'], job['type'], job['label'],
                                                     additional_columns=additional_columns)
            dump_to_cache(processed_df_cache, (training_df, test_df), prefix="cache/labeled_log_cache/")
    else:
        training_log, test_log, additional_columns = prepare_logs(job['split'])
        training_df, test_df = encode_label_logs(training_log, test_log, job['encoding'], job['type'], job['label'],
                                                 additional_columns=additional_columns)
    return training_df, test_df


def run_by_type(training_df: DataFrame, test_df: DataFrame, job: dict) -> (dict, dict):
    """runs the specified training/evaluation run

    :param training_df: training DataFrame
    :param test_df: testing DataFrame
    :param job: job configuration
    :return: results and predictive_model split

    """
    model_split = None

    if job['incremental_train']['base_model'] is not None:
        job['type'] = UPDATE

    start_time = time.time()
    if job['type'] == CLASSIFICATION:
        results, model_split = classification(training_df, test_df, job)
    elif job['type'] == REGRESSION:
        results, model_split = regression(training_df, test_df, job)
    elif job['type'] == TIME_SERIES_PREDICTION:
        results, model_split = time_series_prediction(training_df, test_df, job)
    elif job['type'] == LABELLING:
        results = _label_task(training_df)
    elif job['type'] == UPDATE:
        results, model_split = update_and_test(training_df, test_df, job)
    else:
        raise ValueError("Type not supported", job['type'])

    if job['type'] == CLASSIFICATION:
        save_result(results, job, start_time)

    print("End job {}, {} .".format(job['type'], get_run(job)))
    print("\tResults {} .".format(results))
    return results, model_split


def runtime_calculate(run_log: list, model: dict) -> dict:
    """calculate the predictive_model's score for runtime tasks

    :param run_log: run dataset
    :param model: predictive_model dictionary
    :return: runtime results

    """
    run_df = encode_label_log(run_log, model['encoding'], model['type'], model['label'])
    if model['type'] == CLASSIFICATION:
        results = classification_single_log(run_df, model)
    elif model['type'] == REGRESSION:
        results = regression_single_log(run_df, model)
    elif model['type'] == TIME_SERIES_PREDICTION:
        results = time_series_prediction_single_log(run_df, model)
    else:
        raise ValueError("Type not supported", model['type'])
    print("End job {}, {} . Results {}".format(model['type'], get_run(model), results))
    return results


def get_run(job: dict) -> str:
    """defines the job's identity

    returns a string indicating the job configuration in an unique way

    :param job: job configuration
    :return: job's identity string
    """
    if job['type'] == LABELLING:
        return job['encoding'].method + '_' + job['label'].type
    return job['method'] + '_' + job['encoding'].method + '_' + job['clustering'] + '_' + job['label'].type


def _label_task(input_dataframe: DataFrame) -> dict:
    """calculates the distribution of labels in the data frame

    :return: Dict of string and int {'label1': label1_count, 'label2': label2_count}

    """
    # Stupid but it works
    # True must be turned into 'true'
    json_value = input_dataframe.label.value_counts().to_json()
    return json.loads(json_value)
