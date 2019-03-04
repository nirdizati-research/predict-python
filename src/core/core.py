import json
import os
import time

from pandas import DataFrame

from src.encoding.common import encode_label_log, encode_label_logs
from src.evaluation.models import Evaluation
from src.jobs.models import JobTypes, Job
from src.predictive_model.classification.classification import classification_single_log, update_and_test, \
    classification
from src.predictive_model.models import PredictiveModelTypes
from src.predictive_model.regression.regression import regression, regression_single_log
from src.predictive_model.time_series_prediction.time_series_prediction import time_series_prediction_single_log, \
    time_series_prediction
from src.split.splitting import prepare_logs
from src.cache.cache import load_from_cache, dump_to_cache, get_digested
from src.utils.file_service import save_result


def calculate(job: Job) -> (dict, dict):
    """main entry point for calculations

    encodes the logs based on the given configuration and runs the selected task
    :param job: job configuration
    :return: results and predictive_model split

    """
    print("Start job {} with {}".format(job.type, get_run(job)))
    training_df, test_df = get_encoded_logs(job)
    results, model_split = run_by_type(training_df, test_df, job)
    return results, model_split


def get_encoded_logs(job: Job, use_cache: bool = True) -> (DataFrame, DataFrame):
    """returns the encoded logs

    returns the training and test DataFrames encoded using the given job configuration, loading from cache if possible
    :param job: job configuration
    :param use_cache: load or not saved datasets from cache
    :return: training and testing DataFrame

    """
    print('\tGetting Dataset')
    if use_cache:
        if LabelledLogs.objects.filter(split=job.split,
                                       encoding=job.encoding,
                                       labelling=job.labelling).exists():
            training_df, test_df = get_labelled_logs(job)

        else:
            if job.split.train_log is not None and \
                job.split.test_log is not None and \
                LoadedLog.objects.filter(train_log=job.split.train_log.path,
                                         test_log=job.split.test_log.path).exists():
                training_log, test_log, additional_columns = get_loaded_logs(job.split)

            else:
                training_log, test_log, additional_columns = prepare_logs(job.split)
                put_loaded_logs(job.split, training_log, test_log, additional_columns)

            training_df, test_df = encode_label_logs(training_log, test_log, job.encoding, job.type, job.labelling,
                                                     additional_columns=additional_columns, split_id=job.split.id)
            put_labelled_logs(job, training_df, test_df)
    else:
        training_log, test_log, additional_columns = prepare_logs(job.split)
        training_df, test_df = encode_label_logs(training_log, test_log, job, additional_columns=additional_columns)
    return training_df, test_df


def run_by_type(training_df: DataFrame, test_df: DataFrame, job: Job) -> (dict, dict):
    """runs the specified training/evaluation run

    :param training_df: training DataFrame
    :param test_df: testing DataFrame
    :param job: job configuration
    :return: results and predictive_model split

    """
    model_split = None

    if job['incremental_train']['base_model'] is not None:
        job['type'] = JobTypes.UPDATE.value

    start_time = time.time()
    if job.type == JobTypes.PREDICTION.value:
        if job.predictive_model.type == PredictiveModelTypes.CLASSIFICATION.value:
            results, model_split = classification(training_df, test_df, job)
        elif job.predictive_model.type == PredictiveModelTypes.REGRESSION.value:
            results, model_split = regression(training_df, test_df, job)
        elif job.predictive_model.type == PredictiveModelTypes.TIME_SERIES_PREDICTION.value:
            results, model_split = time_series_prediction(training_df, test_df, job)
    elif job.type == JobTypes.LABELLING.value:
        results = _label_task(training_df)
    elif job.type == JobTypes.UPDATE.value:
        results, model_split = update_and_test(training_df, test_df, job)
    else:
        raise ValueError("Type not supported", job['type'])

    #TODO: integrateme
    Job.evaluation = Evaluation.init(job.predictive_model.type, results)

    if job.type == PredictiveModelTypes.CLASSIFICATION.value:
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
    if model['type'] == PredictiveModelTypes.CLASSIFICATION.value:
        results = classification_single_log(run_df, model)
    elif model['type'] == PredictiveModelTypes.REGRESSION.value:
        results = regression_single_log(run_df, model)
    elif model['type'] == PredictiveModelTypes.TIME_SERIES_PREDICTION.value:
        results = time_series_prediction_single_log(run_df, model)
    else:
        raise ValueError("Type not supported", model['type'])
    print("End job {}, {} . Results {}".format(model['type'], get_run(model), results))
    return results


def get_run(job: Job) -> str:
    """defines the job's identity

    returns a string indicating the job configuration in an unique way

    :param job: job configuration
    :return: job's identity string
    """
    if job.labelling.type == JobTypes.LABELLING.value:
        return job.encoding.data_encoding + '_' + job.labelling.type
    return job.type + '_' + \
           job.encoding.data_encoding + '_' + \
           job.clustering.__class__.__name__ + '_' + \
           job.labelling.type


def _label_task(input_dataframe: DataFrame) -> dict:
    """calculates the distribution of labels in the data frame

    :return: Dict of string and int {'label1': label1_count, 'label2': label2_count}

    """
    # Stupid but it works
    # True must be turned into 'true'
    json_value = input_dataframe.label.value_counts().to_json()
    return json.loads(json_value)
