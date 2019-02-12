from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.externals import joblib

from core.clustering import Clustering
from core.common import get_method_config
from core.constants import RNN
from core.nn.rnn_time_series_predictor import RNNTimeSeriesPredictor
from utils.result_metrics import calculate_results_classification

pd.options.mode.chained_assignment = None


def time_series_prediction(train_df: DataFrame, test_df: DataFrame, job: dict):
    train_data, test_data = _drop_columns(train_df, test_df)

    model_split = _train(job, train_data, _choose_classifier(job))
    results_df, nlevenshtein = _test(model_split, test_data, evaluation=True)

    results = _prepare_results(results_df, nlevenshtein)

    # TODO save model more wisely
    model_split['type'] = job['clustering']

    return results, model_split


def time_series_prediction_single_log(data: DataFrame, model: dict):
    results = dict()
    split = model['split']
    results['input'] = data

    # TODO load model more wisely
    model_split = dict()
    model_split['clusterer'] = joblib.load(split['clusterer_path'])
    model_split['time_series_predictor'] = joblib.load(split['model_path'])
    result, _ = _test(model_split, data, evaluation=False)
    results['prediction'] = result['predicted']
    return results


def _train(job: dict, train_data: DataFrame, time_series_predictor: Any) -> dict:
    clusterer = Clustering(job)
    models = dict()

    clusterer.fit(train_data)

    train_data = clusterer.cluster_data(train_data)

    for cluster in range(clusterer.n_clusters):

        x = train_data[cluster]
        if not x.empty:
            time_series_predictor.fit(x)

            models[cluster] = time_series_predictor

    return {'clusterer': clusterer, 'time_series_predictor': models}


def _test(model_split: dict, data: DataFrame, evaluation: bool) -> (dict, float):
    clusterer = model_split['clusterer']
    time_series_predictor = model_split['time_series_predictor']

    test_data = clusterer.cluster_data(data)

    results_df = DataFrame()

    non_empty_clusters = clusterer.n_clusters

    nlevenshtein_distances = []

    for cluster in range(clusterer.n_clusters):
        x = test_data[cluster]
        if x.empty:
            non_empty_clusters -= 1
        else:
            if evaluation:
                predictions = time_series_predictor[cluster].predict(x)

            x['predicted'] = time_series_predictor[cluster].predict(x)

            results_df = results_df.append(x)

    return results_df, nlevenshtein


def _prepare_results(df: DataFrame, auc: int) -> dict:
    actual = df['label'].values
    predicted = df['predicted'].values

    row = calculate_results_classification(actual, predicted)
    row['auc'] = auc
    return row


def _drop_columns(train_df: DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame):
    train_df = train_df.drop(['trace_id', 'label'], 1)
    test_df = test_df.drop(['trace_id', 'label'], 1)
    return train_df, test_df


def _choose_classifier(job: dict) -> Any:
    method, config = get_method_config(job)
    print("Using method {} with config {}".format(method, config))
    if method == RNN:
        config['encoding'] = job['encoding'][0]
        time_series_predictor = RNNTimeSeriesPredictor(**config)
    else:
        raise ValueError("Unexpected time series prediction method {}".format(method))
    return time_series_predictor
