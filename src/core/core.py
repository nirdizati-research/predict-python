import json
import logging
import time
from datetime import timedelta

import pandas as pd
from pandas import DataFrame
from pm4py.objects.log.log import EventLog
from sklearn.model_selection import train_test_split

from src.evaluation.models import Evaluation
from src.jobs.models import JobTypes, Job
from src.labelling.models import Labelling
from src.predictive_model.common import ModelActions, MODEL
from src.predictive_model.models import PredictiveModels
from src.utils.django_orm import duplicate_orm_row
from src.utils.event_attributes import get_additional_columns
from src.clustering.clustering import init_clusterer
from src.encoding.common import encode_label_logs, data_encoder_decoder, get_encoded_logs

logger = logging.getLogger(__name__)


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
        clusterer = init_clusterer(job.clustering, training_df)
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
        job.save()
    elif job.type == JobTypes.LABELLING.value:
        # job.labelling = duplicate_orm_row(job.labelling) #todo: replace with simple CREATE
        job.labelling = Labelling.objects.create(
            type=job.labelling.type,
            attribute_name=job.labelling.attribute_name,
            threshold_type=job.labelling.threshold_type,
            threshold=job.labelling.threshold
        ) #todo: futurebug if object changes
        job.labelling.results = results
        job.labelling.save()
        job.save()

    # if job.type == PredictiveModels.CLASSIFICATION.value: #todo this is an old workaround I should remove this
    #     save_result(results, job, start_time)

    logger.info("End job {}, {} .".format(job.type, get_run(job)))
    logger.info("\tResults {} .".format(results))
    return results, model_split


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
