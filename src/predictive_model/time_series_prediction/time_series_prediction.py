"""
time series prediction methods and functionalities
"""

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import clone
from sklearn.externals import joblib

from src.clustering.clustering import Clustering
from src.core.common import get_method_config
from src.jobs.models import Job
from src.predictive_model.time_series_prediction import TimeSeriesPredictorMixin
from src.predictive_model.time_series_prediction.custom_time_series_prediction_models import RNNTimeSeriesPredictor
from src.predictive_model.time_series_prediction.models import TimeSeriesPredictionMethods
from src.utils.result_metrics import calculate_results_time_series_prediction, \
    calculate_nlevenshtein

pd.options.mode.chained_assignment = None


def time_series_prediction(training_df: DataFrame, test_df: DataFrame, clusterer: Clustering, job: Job) -> (dict, dict):
    """main time series prediction entry point

    train and tests the time series predictor using the provided data

    :param clusterer:
    :param training_df: training DataFrame
    :param test_df: testing DataFrame
    :param job: job configuration
    :return: predictive_model scores and split

    """
    train_data, test_data = _drop_columns(training_df, test_df)

    model_split = _train(train_data, _choose_time_series_predictor(job), clusterer)
    results_df, nlevenshtein = _test(model_split, test_data, evaluation=True)

    results = _prepare_results(results_df, nlevenshtein)

    # TODO save predictive_model more wisely
    model_split['type'] = job['clustering']

    return results, model_split


def time_series_prediction_single_log(input_df: DataFrame, model: dict) -> dict:
    """single log time series prediction

    time series predicts a single log using the provided TODO: complete

    :param input_df: input DataFrame
    :param model: TODO: complete
    :return: predictive_model scores

    """
    results = dict()
    split = model['split']
    results['input'] = input_df

    # TODO load predictive_model more wisely
    model_split = dict()
    model_split['clusterer'] = joblib.load(split['clusterer_path'])
    model_split['time_series_predictor'] = joblib.load(split['model_path'])
    result, _ = _test(model_split, input_df, evaluation=False)
    results['prediction'] = result['predicted']
    return results


def _train(train_data: DataFrame, time_series_predictor: Any, clusterer: Clustering) -> dict:
    models = dict()

    train_data = clusterer.cluster_data(train_data)

    for cluster in range(clusterer.n_clusters):

        cluster_train_df = train_data[cluster]
        if not cluster_train_df.empty:
            time_series_predictor.fit(cluster_train_df)

            models[cluster] = time_series_predictor
            time_series_predictor = clone(time_series_predictor, safe=False)
    return {'clusterer': clusterer, 'time_series_predictor': models}


def _test(model_split: dict, data: DataFrame, evaluation: bool) -> (DataFrame, float):
    clusterer = model_split['clusterer']
    time_series_predictor = model_split['time_series_predictor']

    test_data = clusterer.cluster_data(data)

    results_df = DataFrame()

    non_empty_clusters = clusterer.n_clusters

    nlevenshtein_distances = []

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = test_data[cluster]
        if cluster_test_df.empty:
            non_empty_clusters -= 1
        else:
            if evaluation:
                predictions = time_series_predictor[cluster].predict(cluster_test_df)

                nlevenshtein = calculate_nlevenshtein(cluster_test_df.values, predictions)
                nlevenshtein_distances.append(nlevenshtein)
            temp_actual = cluster_test_df.values.tolist()
            cluster_test_df['predicted'] = time_series_predictor[cluster].predict(cluster_test_df).tolist()
            cluster_test_df['actual'] = temp_actual

            results_df = results_df.append(cluster_test_df)

    nlevenshtein = float(np.mean(nlevenshtein_distances))

    return results_df, nlevenshtein


def _prepare_results(df: DataFrame, nlevenshtein: float) -> dict:
    actual = np.array(df['actual'].values.tolist())
    predicted = np.array(df['predicted'].values.tolist())
    row = calculate_results_time_series_prediction(actual, predicted)
    row['nlevenshtein'] = nlevenshtein
    return row


def _drop_columns(train_df: DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame):
    train_df = train_df.drop(['trace_id', 'label'], 1)
    test_df = test_df.drop(['trace_id', 'label'], 1)
    return train_df, test_df


def _choose_time_series_predictor(job: Job) -> TimeSeriesPredictorMixin:
    method, config = get_method_config(job)
    print("Using method {} with config {}".format(method, config))
    if method == TimeSeriesPredictionMethods.RNN.value:
        config['encoding'] = job.encoding.value_encoding
        time_series_predictor = RNNTimeSeriesPredictor(**config)
    else:
        raise ValueError("Unexpected time series prediction method {}".format(method))
    return time_series_predictor
