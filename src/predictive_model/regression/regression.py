"""
regression methods and functionalities
"""

import pandas as pd
from pandas import DataFrame
from sklearn import clone
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.clustering.clustering import Clustering
from src.core.common import get_method_config
from src.core.constants import LINEAR, RANDOM_FOREST, LASSO, XGBOOST, NN
from src.predictive_model.nn_models import NNRegressor
from src.utils.result_metrics import calculate_results_regression

pd.options.mode.chained_assignment = None


def regression(training_df: DataFrame, test_df: DataFrame, job: dict) -> (dict, dict):
    """main regression entry point

    train and tests the regressor using the provided data

    :param training_df: training DataFrame
    :param test_df: testing DataFrame
    :param job: job configuration
    :return: predictive_model scores and split

    """
    train_data, test_data = _prep_data(training_df, test_df)

    model_split = _train(job, train_data, _choose_regressor(job))
    results_df = _test(model_split, test_data)

    results = calculate_results_regression(results_df, job['label'])

    # TODO save predictive_model more wisely
    model_split['type'] = job['clustering']

    return results, model_split


def regression_single_log(input_df: DataFrame, model: dict) -> DataFrame:
    """single log regression

    classifies a single log using the provided TODO: complete

    :param input_df: input DataFrame
    :param model: TODO: complete
    :return: predictive_model scores

    """
    split = model['split']
    input_df = input_df.drop([col for col in ['label', 'remaining_time', 'trace_id'] if col in input_df.columns], 1)

    # TODO load predictive_model more wisely
    model_split = dict()
    model_split['clusterer'] = joblib.load(split['clusterer_path'])
    model_split['regressor'] = joblib.load(split['model_path'])
    results_df = _test(model_split, input_df)
    return results_df


def _train(job: dict, train_data: DataFrame, regressor: RegressorMixin) -> dict:
    clusterer = Clustering(job)
    models = dict()

    clusterer.fit(train_data.drop('label', 1))

    train_data = clusterer.cluster_data(train_data)

    for cluster in range(clusterer.n_clusters):

        cluster_train_df = train_data[cluster]
        if not cluster_train_df.empty:
            cluster_targets_df = cluster_train_df['label']
            regressor.fit(cluster_train_df.drop('label', 1), cluster_targets_df.values.ravel())

            models[cluster] = regressor
            try:
                regressor = clone(regressor)
            except TypeError:
                regressor = clone(regressor, safe=False)

    return {'clusterer': clusterer, 'regressor': models}


def _test(model_split: dict, data: DataFrame) -> DataFrame:
    clusterer = model_split['clusterer']
    regressor = model_split['regressor']

    test_data = clusterer.cluster_data(data)

    results_df = DataFrame()

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = test_data[cluster]
        if not cluster_test_df.empty:
            cluster_test_df['predicted'] = regressor[cluster].predict(cluster_test_df.drop('label', 1))
            results_df = results_df.append(cluster_test_df)
    return results_df


def _prep_data(training_df: DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame):
    train_data = training_df
    test_data = test_df

    test_data = test_data.drop(['trace_id'], 1)
    train_data = train_data.drop('trace_id', 1)
    return train_data, test_data


def _choose_regressor(job: dict) -> RegressorMixin:
    method, config = get_method_config(job)
    print("Using method {} with config {}".format(method, config))
    if method == LINEAR:
        regressor = LinearRegression(**config)
    elif method == RANDOM_FOREST:
        regressor = RandomForestRegressor(**config)
    elif method == LASSO:
        regressor = Lasso(**config)
    elif method == XGBOOST:
        regressor = XGBRegressor(**config)
    elif method == NN:
        config['encoding'] = job['encoding'][0]
        regressor = NNRegressor(**config)
    else:
        raise ValueError("Unexpected regression method {}".format(method))
    return regressor
