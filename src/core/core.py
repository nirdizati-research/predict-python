import json
import logging
import time
import pandas as pd
from datetime import timedelta
from enum import Enum

from pandas import DataFrame
from pm4py.objects.log.log import EventLog
from sklearn.model_selection import train_test_split

from src.cache.cache import get_labelled_logs, get_loaded_logs, \
    put_loaded_logs, put_labelled_logs
from src.cache.models import LabelledLog, LoadedLog
from src.clustering.clustering import Clustering
from src.encoding.common import encode_label_logs, data_encoder_decoder
from src.evaluation.models import Evaluation
from src.jobs.models import JobTypes, Job
from src.logs.log_service import create_log
from src.predictive_model.classification import classification
from src.predictive_model.models import PredictiveModels
from src.predictive_model.regression import regression
from src.predictive_model.time_series_prediction import time_series_prediction
from src.split.models import SplitTypes, Split
from src.split.splitting import get_train_test_log
from src.utils.django_orm import duplicate_orm_row
from src.utils.event_attributes import get_additional_columns

logger = logging.getLogger(__name__)


class ModelActions (Enum):
    PREDICT = 'predict'
    PREDICT_PROBA = 'predict_proba'
    UPDATE_AND_TEST = 'update_and_test'
    BUILD_MODEL_AND_TEST = 'build_model_and_test'


MODEL = {
    PredictiveModels.CLASSIFICATION.value: {
        ModelActions.PREDICT.value: classification.predict,
        ModelActions.PREDICT_PROBA.value: classification.predict_proba,
        ModelActions.UPDATE_AND_TEST.value: classification.update_and_test,
        ModelActions.BUILD_MODEL_AND_TEST.value: classification.classification
    },
    PredictiveModels.REGRESSION.value: {
        ModelActions.PREDICT.value: regression.predict,
        ModelActions.BUILD_MODEL_AND_TEST.value: regression.regression
    },
    PredictiveModels.TIME_SERIES_PREDICTION.value: {
        ModelActions.PREDICT.value: time_series_prediction.predict,
        ModelActions.BUILD_MODEL_AND_TEST.value: time_series_prediction.time_series_prediction
    }
}


def calculate(job: Job) -> (dict, dict): #TODO dd filter for 'valid' configurations
    """main entry point for calculations

    encodes the logs based on the given configuration and runs the selected task
    :param job: job configuration
    :return: results and predictive_model split

    """
    logger.info("Start job {} with {}".format(job.type, get_run(job)))
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
    logger.info('\tGetting Dataset')

    if use_cache and \
        (job.predictive_model is not None and
         job.predictive_model.predictive_model != PredictiveModels.TIME_SERIES_PREDICTION.value):

        if LabelledLog.objects.filter(split=job.split,
                                      encoding=job.encoding,
                                      labelling=job.labelling).exists():
            try:
                training_df, test_df = get_labelled_logs(job)
            except FileNotFoundError: #cache invalidation
                LabelledLog.objects.filter(split=job.split,
                                           encoding=job.encoding,
                                           labelling=job.labelling).delete()
                logger.info('\t\tError pre-labeled cache invalidated!')
                return get_encoded_logs(job, use_cache)
        else:
            if job.split.train_log is not None and \
               job.split.test_log is not None and \
               LoadedLog.objects.filter(split=job.split).exists():
                try:
                    training_log, test_log, additional_columns = get_loaded_logs(job.split)
                except FileNotFoundError:  # cache invalidation
                    LoadedLog.objects.filter(split=job.split).delete()
                    logger.info('\t\tError pre-loaded cache invalidated!')
                    return get_encoded_logs(job, use_cache)
            else:
                training_log, test_log, additional_columns = get_train_test_log(job.split)
                if job.split.type == SplitTypes.SPLIT_SINGLE.value:
                    search_for_already_existing_split = Split.objects.filter(
                        type=SplitTypes.SPLIT_DOUBLE.value,
                        original_log=job.split.original_log,
                        test_size=job.split.test_size,
                        splitting_method=job.split.splitting_method
                    )
                    if len(search_for_already_existing_split) >= 1:
                        job.split = search_for_already_existing_split[0]
                        job.split.save()
                        job.save()
                        return get_encoded_logs(job, use_cache=use_cache)
                    else:
                        job.split = duplicate_orm_row(Split.objects.filter(pk=job.split.pk)[0])
                        job.split.type = SplitTypes.SPLIT_DOUBLE.value
                        train_name = 'SPLITTED_' + job.split.original_log.name.split('.')[0] + '_0-' + str(int(100 - (job.split.test_size * 100)))
                        job.split.train_log = create_log(
                            EventLog(training_log),
                            train_name + '.xes'
                        )
                        test_name = 'SPLITTED_' + job.split.original_log.name.split('.')[0] + '_' + str(int(100 - (job.split.test_size * 100))) + '-100'
                        job.split.test_log = create_log(
                            EventLog(test_log),
                            test_name + '.xes'
                        )
                        job.split.additional_columns = str(train_name + test_name)  # TODO: find better naming policy
                        job.split.save()

                put_loaded_logs(job.split, training_log, test_log, additional_columns)

            training_df, test_df = encode_label_logs(
                training_log,
                test_log,
                job,
                additional_columns=additional_columns)
            put_labelled_logs(job, training_df, test_df)
    else:
        training_log, test_log, additional_columns = get_train_test_log(job.split)
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

    start_time = time.time()
    if job.type == JobTypes.PREDICTION.value:
        clusterer = _init_clusterer(job.clustering, training_df)
        results, model_split = MODEL[job.predictive_model.predictive_model][ModelActions.BUILD_MODEL_AND_TEST.value](training_df, test_df, clusterer, job)
    elif job.type == JobTypes.LABELLING.value:
        results = _label_task(training_df)
    elif job.type == JobTypes.UPDATE.value:
        results, model_split = MODEL[job.predictive_model.predictive_model][ModelActions.UPDATE_AND_TEST.value](training_df, test_df, job)
    else:
        raise ValueError("Type {} not supported".format(job.type))

    # TODO: integrateme
    if job.type != JobTypes.LABELLING.value:
        results['elapsed_time'] = timedelta(seconds=time.time() - start_time) #todo find better place for this
        if job.predictive_model.predictive_model == PredictiveModels.REGRESSION.value:
            job.evaluation = Evaluation.init(
                job.predictive_model.predictive_model,
                results
            )
        elif job.predictive_model.predictive_model == PredictiveModels.CLASSIFICATION.value:
            job.evaluation = Evaluation.init(
                job.predictive_model.predictive_model,
                results,
                len(set(test_df['label'])) <= 2
            )
        elif job.predictive_model.predictive_model == PredictiveModels.TIME_SERIES_PREDICTION.value:
            job.evaluation = Evaluation.init(
                job.predictive_model.predictive_model,
                results
            )
        job.evaluation.save()
    elif job.type == JobTypes.LABELLING.value:
        job.labelling = duplicate_orm_row(job.labelling)
        job.labelling.results = results
        job.labelling.save()
        job.save()

    # if job.type == PredictiveModels.CLASSIFICATION.value: #todo this is an old workaround I should remove this
    #     save_result(results, job, start_time)

    logger.info("End job {}, {} .".format(job.type, get_run(job)))
    logger.info("\tResults {} .".format(results))
    return results, model_split


def _init_clusterer(clustering: Clustering, train_data: DataFrame):
    clusterer = Clustering(clustering)
    clusterer.fit(train_data.drop(['trace_id', 'label'], 1))
    return clusterer


def runtime_calculate(job: Job) -> dict:
    """calculate the prediction for traces in the uncompleted logs

    :param job: job idctionary
    :return: runtime results
    """

    training_df, test_df = get_encoded_logs(job)
    data_df = pd.concat([training_df,test_df])
    results = MODEL[job.predictive_model.predictive_model][ModelActions.PREDICT.value](job, data_df)
    logger.info("End {} job {}, {} . Results {}".format('runtime', job.predictive_model.predictive_model, get_run(job), results))
    return results


def replay_prediction_calculate(job: Job, log) -> (dict, dict):
    """calculate the prediction for the log coming from replayers

    :param job: job dictionary
    :param log: log model
    :return: runtime results
    """
    additional_columns = get_additional_columns(log)
    data_df, _ = train_test_split(log, test_size=0, shuffle=False)
    data_df, _ = encode_label_logs(data_df, EventLog(), job, additional_columns)
    results = MODEL[job.predictive_model.predictive_model][ModelActions.PREDICT.value](job, data_df)
    logger.info("End {} job {}, {} . Results {}".format('runtime', job.predictive_model.predictive_model, get_run(job), results))
    results_dict = dict(zip(data_df['trace_id'], list(map(int, results))))
    events_for_trace = dict()
    data_encoder_decoder(job, data_df, EventLog())
    return results_dict, events_for_trace


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
