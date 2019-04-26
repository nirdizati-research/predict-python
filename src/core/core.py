import json
import time

from pandas import DataFrame
from pm4py.objects.log.log import EventLog

from src.cache.cache import get_labelled_logs, get_loaded_logs, \
    put_loaded_logs, put_labelled_logs
from src.cache.models import LabelledLog, LoadedLog
from src.clustering.clustering import Clustering
from src.encoding.common import encode_label_log, encode_label_logs
from src.evaluation.models import Evaluation
from src.jobs.models import JobTypes, Job, ModelType
from src.logs.log_service import create_log
from src.predictive_model.classification.classification import classification_single_log, update_and_test, \
    classification
from src.predictive_model.models import PredictiveModels
from src.predictive_model.regression.regression import regression, regression_single_log
from src.predictive_model.time_series_prediction.time_series_prediction import time_series_prediction_single_log, \
    time_series_prediction
from src.split.models import SplitTypes
from src.split.splitting import prepare_logs
from src.utils.django_orm import duplicate_orm_row
from src.utils.file_service import save_result


def calculate(job: Job) -> (dict, dict): #TODO dd filter for 'valid' configurations
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
        if LabelledLog.objects.filter(split=job.split,
                                      encoding=job.encoding,
                                      labelling=job.labelling).exists():
            try:
                training_df, test_df = get_labelled_logs(job)
            except FileNotFoundError:  # cache invalidation
                LabelledLog.objects.filter(split=job.split,
                                           encoding=job.encoding,
                                           labelling=job.labelling).delete()
                return get_encoded_logs(job, use_cache)
        else:
            if job.split.train_log is not None and \
                job.split.test_log is not None and \
                LoadedLog.objects.filter(train_log=job.split.train_log.path,
                                         test_log=job.split.test_log.path).exists():
                training_log, test_log, additional_columns = get_loaded_logs(job.split)

            else:
                training_log, test_log, additional_columns = prepare_logs(job.split)
                if job.split.type == SplitTypes.SPLIT_SINGLE.value:
                    job.split = duplicate_orm_row(job.split)
                    job.split.type = SplitTypes.SPLIT_DOUBLE.value
                    train_name = '0-' + str(int(100 - (job.split.test_size * 100)))
                    job.split.train_log = create_log(
                        EventLog(training_log),
                        train_name + '.xes'
                    )
                    test_name = str(int(100 - (job.split.test_size * 100))) + '-100'
                    job.split.test_log = create_log(
                        EventLog(test_log),
                        test_name + '.xes'
                    )
                    job.split.additional_columns = str(train_name + test_name)  # TODO: find better naming policy
                    job.save()

                put_loaded_logs(job.split, training_log, test_log, additional_columns)

            training_df, test_df = encode_label_logs(
                training_log,
                test_log,
                job,
                additional_columns=additional_columns)
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

    # TODO fixme this needs to be fixed in the interface
    # if job['incremental_train']['base_model'] is not None:
    #     job['type'] = JobTypes.UPDATE.value

    start_time = time.time()
    if job.type == JobTypes.PREDICTION.value:
        clusterer = _init_clusterer(job.clustering, training_df)
        if job.predictive_model.predictive_model == PredictiveModels.CLASSIFICATION.value:
            results, model_split = classification(training_df, test_df, clusterer, job)
        elif job.predictive_model.predictive_model == PredictiveModels.REGRESSION.value:
            results, model_split = regression(training_df, test_df, clusterer, job)
        elif job.predictive_model.predictive_model == PredictiveModels.TIME_SERIES_PREDICTION.value:
            results, model_split = time_series_prediction(training_df, test_df, clusterer, job)
    elif job.type == JobTypes.LABELLING.value:
        results = _label_task(training_df)
    elif job.type == JobTypes.UPDATE.value:
        results, model_split = update_and_test(training_df, test_df, job)
    else:
        raise ValueError("Type {} not supported".format(job.type))

    # TODO: integrateme
    if job.type != JobTypes.LABELLING.value:
        if job.predictive_model.predictive_model == PredictiveModels.REGRESSION.value:
            job.evaluation = Evaluation.init(
                job.predictive_model.predictive_model,
                results
            )
        elif job.predictive_model.predictive_model == PredictiveModels.CLASSIFICATION.value:
            job.evaluation = Evaluation.init(
                job.predictive_model.predictive_model,
                results,
                len(model_split[ModelType.CLASSIFIER.value][0].classes_) <= 2
            )
        job.save()

    if job.type == PredictiveModels.CLASSIFICATION.value:
        save_result(results, job, start_time)

    print("End job {}, {} .".format(job.type, get_run(job)))
    print("\tResults {} .".format(results))
    return results, model_split


def _init_clusterer(clustering: Clustering, train_data: DataFrame):
    clusterer = Clustering(clustering)
    clusterer.fit(train_data.drop(['trace_id', 'label'], 1))
    return clusterer


def runtime_calculate(run_log: list, model: dict) -> dict:
    """calculate the predictive_model's score for runtime tasks

    :param run_log: run dataset
    :param model: predictive_model dictionary
    :return: runtime results

    """
    run_df = encode_label_log(run_log, model['encoding'], model['type'], model['label'])
    if model['type'] == PredictiveModels.CLASSIFICATION.value:
        results = classification_single_log(run_df, model)
    elif model['type'] == PredictiveModels.REGRESSION.value:
        results = regression_single_log(run_df, model)
    elif model['type'] == PredictiveModels.TIME_SERIES_PREDICTION.value:
        results = time_series_prediction_single_log(run_df, model)
    else:
        raise ValueError("Type {} not supported".format(model['type']))
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
    return '_'.join([job.type, job.encoding.data_encoding, job.clustering.__class__.__name__, job.labelling.type])


def _label_task(input_dataframe: DataFrame) -> dict:
    """calculates the distribution of labels in the data frame

    :return: Dict of string and int {'label1': label1_count, 'label2': label2_count}

    """
    # Stupid but it works
    # True must be turned into 'true'
    json_value = input_dataframe.label.value_counts().to_json()
    return json.loads(json_value)
