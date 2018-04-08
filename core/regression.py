from math import sqrt

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from core.constants import KMEANS, LINEAR, RANDOM_FOREST, LASSO

pd.options.mode.chained_assignment = None


def regression(training_df, test_df, job):
    regressor = __choose_regressor(job['method'])

    train_data, test_data, original_test_data = prep_data(training_df, test_df)

    if job['clustering'] == KMEANS:
        results_df = kmeans_clustering(original_test_data, train_data, regressor)
    else:
        results_df = no_clustering(original_test_data, train_data, test_data, regressor)

    results = prepare_results(results_df)
    return results


def kmeans_clustering(original_test_data, train_data, regressor):
    estimator = KMeans(n_clusters=3)
    estimator.fit(train_data)

    original_cluster_lists = {
        i: original_test_data.iloc[np.where(estimator.predict(original_test_data.drop('trace_id', 1)) == i)[0]]
        for i in range(estimator.n_clusters)}
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    result_data = None
    for cluster_list in cluster_lists:
        original_test_clustered_data = original_cluster_lists[cluster_list]
        if original_test_clustered_data.shape[0] == 0:
            pass
        else:
            clustered_train_data = cluster_lists[cluster_list]
            clustered_test_data = original_cluster_lists[cluster_list]
            clustered_test_data = clustered_test_data.drop(['trace_id', 'remaining_time'], 1)

            y = clustered_train_data['remaining_time']
            clustered_train_data = clustered_train_data.drop('remaining_time', 1)

            regressor.fit(clustered_train_data, y)
            original_test_clustered_data['prediction'] = regressor.predict(clustered_test_data)
            if result_data is None:
                result_data = original_test_clustered_data
            else:
                result_data = result_data.append(original_test_clustered_data)
    return result_data


def no_clustering(original_test_data, train_data, test_data, regressor):
    y = train_data['remaining_time']
    train_data = train_data.drop('remaining_time', 1)
    regressor.fit(train_data, y)
    original_test_data['prediction'] = regressor.predict(test_data)
    return original_test_data


def prepare_results(df):
    # TODO are remaining time in seconds or hours?
    df['remaining_time'] = df['remaining_time'] / 3600
    df['prediction'] = df['prediction'] / 3600
    rmse = sqrt(mean_squared_error(df['remaining_time'], df['prediction']))
    mae = mean_absolute_error(df['remaining_time'], df['prediction'])
    rscore = metrics.r2_score(df['remaining_time'], df['prediction'])

    row = {'rmse': rmse, 'mae': mae, 'rscore': rscore}
    return row


def prep_data(training_df, test_df):
    train_data = training_df.drop('elapsed_time', 1)
    test_data = test_df.drop('elapsed_time', 1)

    original_test_data = test_data

    test_data = test_data.drop(['trace_id', 'remaining_time'], 1)
    train_data = train_data.drop('trace_id', 1)
    return train_data, test_data, original_test_data


def __choose_regressor(regression_type: str):
    regressor = None
    if regression_type == LINEAR:
        regressor = LinearRegression(fit_intercept=True)
    elif regression_type == RANDOM_FOREST:
        regressor = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
    elif regression_type == LASSO:
        regressor = Lasso(fit_intercept=True, warm_start=True)
    return regressor
