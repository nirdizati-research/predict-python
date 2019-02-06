import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from core.common import get_method_config
from core.constants import KMEANS, LINEAR, RANDOM_FOREST, LASSO, NO_CLUSTER, XGBOOST, NN
from core.nn.nn_regressor import NNRegressor
from utils.result_metrics import calculate_results_regression

pd.options.mode.chained_assignment = None


def regression(training_df: DataFrame, test_df: DataFrame, job: dict):
    regressor = _choose_regressor(job)

    train_data, test_data, original_test_data = _prep_data(training_df, test_df)

    if job['clustering'] == KMEANS:
        results_df, model_split = _kmeans_clustering_train(original_test_data, train_data, regressor, job['kmeans'])
    else:
        results_df, model_split = _no_clustering_train(original_test_data, train_data, test_data, regressor)

    results = calculate_results_regression(results_df, job['label'])
    return results, model_split


def regression_single_log(run_df: DataFrame, model):
    split = model['split']
    results = dict()
    results['label'] = run_df['label']
    run_df = run_df.drop(['label'], 1)
    if 'elapsed_time' in run_df.columns:
        run_df = run_df.drop(['elapsed_time'], 1)
    run_df = run_df.drop('trace_id', 1)

    if split['type'] == NO_CLUSTER:
        regressor = joblib.load(split['model_path'])
        results_data = _no_clustering_test(run_df, run_df, regressor)
    elif split['type'] == KMEANS:
        regressor = joblib.load(split['model_path'])
        estimator = joblib.load(split['estimator_path'])
        results_data = _kmeans_clustering_test(run_df, regressor, estimator)
    else:
        raise ValueError('Unexpected split type {}'.format(split['type']))
    results['prediction'] = results_data['prediction']
    return results


def _kmeans_clustering_train(original_test_data, train_data, regressor, kmeans_dict: dict):
    estimator = KMeans(**kmeans_dict)
    estimator.fit(train_data)
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    models = dict()
    for i, cluster_list in cluster_lists.items():
        clustered_train_data = cluster_list
        if clustered_train_data.shape[0] == 0:
            pass
        else:
            y = clustered_train_data['label']
            clustered_train_data = clustered_train_data.drop('label', 1)

            regressor.fit(clustered_train_data, y)
            models[i] = regressor
    model_split = dict()
    model_split['type'] = KMEANS
    model_split['estimator'] = estimator
    model_split['model'] = models
    return _kmeans_clustering_test(original_test_data, models, estimator, testing=True), model_split


def _kmeans_clustering_test(test_data, regressor, estimator, testing=False):
    drop_list = ['trace_id', 'label'] if testing else ['trace_id']
    test_cluster_lists = {
        i: test_data.iloc[np.where(estimator.predict(test_data.drop('trace_id', 1)) == i)[0]]
        for i in range(estimator.n_clusters)}
    result_data = None
    for i, cluster_list in test_cluster_lists.items():
        original_clustered_test_data = cluster_list
        if original_clustered_test_data.shape[0] == 0:
            pass
        else:
            clustered_test_data = original_clustered_test_data.drop(drop_list, 1)
            original_clustered_test_data['prediction'] = regressor[i].predict(clustered_test_data)
        if result_data is None:
            result_data = original_clustered_test_data
        else:
            result_data = result_data.append(original_clustered_test_data)
    return result_data


def _no_clustering_train(original_test_data, train_data, test_data, regressor):
    y = train_data['label']
    train_data = train_data.drop('label', 1)
    regressor.fit(train_data, y)
    model_split = dict()
    model_split['type'] = NO_CLUSTER
    model_split['model'] = regressor
    return _no_clustering_test(original_test_data, test_data, regressor), model_split


def _no_clustering_test(original_test_data, test_data, regressor):
    original_test_data['prediction'] = regressor.predict(test_data)
    return original_test_data


def _prep_data(training_df: DataFrame, test_df: DataFrame):
    train_data = training_df
    test_data = test_df

    original_test_data = test_data

    test_data = test_data.drop(['trace_id', 'label'], 1)
    train_data = train_data.drop('trace_id', 1)
    return train_data, test_data, original_test_data


def _choose_regressor(job: dict):
    method, config = get_method_config(job)
    print("Using method {} with config {}".format(method, config))
    regressor = None
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
    return regressor
