import random
from math import sqrt

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.externals import joblib

from core.constants import KMEANS, LINEAR, RANDOM_FOREST, LASSO
from django.contrib.admin.templatetags.admin_list import results

pd.options.mode.chained_assignment = None


def regression(test_df, job, model):
    split = model['split']
    test_data, original_test_data = prep_data(test_df)
    if split['type'] == 'single':
        regressor = joblib.load(split['model_path'])
    elif split['type'] == 'double':
        regressor = joblib.load(split['model_path'])
        estimator = joblib.load(split['kmean_path'])
        
    if job['clustering'] == KMEANS:
        results_df = kmeans_clustering(original_test_data, regressor, estimator)
    else:
        results_df = no_clustering(original_test_data, test_data, regressor)

    results = prepare_results(results_df)
    return results

def regression_run(run_df, model):
    split = model['split']
    run_df = run_df.drop(columns = 'elapsed_time')
    if split['type'] == 'single':
        regressor = joblib.load(split['model_path'])
        results = no_clustering_run(run_df, regressor)
    elif split['type'] == 'double':
        regressor = joblib.load(split['model_path'])
        estimator = joblib.load(split['kmean_path']) 
        results = kmeans_run(run_df, regressor, estimator)
    return results

def no_clustering_run(run_df, regressor):
    run_df = run_df.drop('trace_id',1)
    results = regressor.predict(run_df)
    return results

def kmeans_run(run_df, regressor, estimator):
    test_cluster_lists = {
        i: run_df.iloc[np.where(estimator.predict(run_df.drop('trace_id', 1)) == i)[0]]
        for i in range(estimator.n_clusters)}
    results = []
    #print (test_cluster_lists)
    for i, cluster_list in test_cluster_lists.items():
        clustered_test_data = cluster_lists
        if clustered_test_data.shape[0] == 0:
            pass
        else:
            clustered_test_data['result']=model[i].predict(clustered_test_data.drop('trace_id',1))       
    return clustered_test_data['result']

def kmeans_clustering(original_test_data, regressor, estimator):
    print(estimator)
    original_cluster_lists = {
        i: original_test_data.iloc[np.where(estimator.predict(original_test_data.drop(['trace_id', 'remaining_time'], 1)) == i)[0]]
        for i in range(estimator.n_clusters)}
    result_data=None
    for i, cluster_list in original_cluster_lists.items():
        original_test_clustered_data = cluster_list
        if original_test_clustered_data.shape[0] == 0:
            pass
        else:
            clustered_test_data = cluster_list
            clustered_test_data = clustered_test_data.drop(['trace_id', 'remaining_time'], 1)
            
            original_test_clustered_data['prediction'] = regressor[i].predict(clustered_test_data)
            if result_data is None:
                result_data = original_test_clustered_data
            else:
                result_data = result_data.append(original_test_clustered_data)
    return result_data


def no_clustering(original_test_data, test_data, regressor):
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


def prep_data(test_df):
    test_data = test_df.drop('elapsed_time', 1)

    original_test_data = test_data

    test_data = test_data.drop(['trace_id', 'remaining_time'], 1)

    return test_data, original_test_data


def __choose_regressor(regression_type: str):
    regressor = None
    if regression_type == LINEAR:
        regressor = LinearRegression(fit_intercept=True)
    elif regression_type == RANDOM_FOREST:
        regressor = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
    elif regression_type == LASSO:
        regressor = Lasso(fit_intercept=True, warm_start=True)
    return regressor
