import cPickle
from math import sqrt

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from core_services.common import encode_if_needed
from core_services.file_service import make_dir, write_to_path, write_results_to_general, write_results_to_db


def regression(job):
    print 'regression queue'
    print job.method_val()

    encode_if_needed(job)
    regressor = __choose_regressor(job)

    train_data, test_data, original_test_data = prep_data(job)

    make_dir(job.prediction_model_dir())
    make_dir(job.get_results_dir())

    if job.clustering == "kmeans":
        kmeans_clustering(original_test_data, train_data, regressor, job)
    else:
        no_clustering(original_test_data, train_data, test_data, regressor, job)

    calculate_results(job)


def kmeans_clustering(original_test_data, train_data, regressor, job):
    estimator = KMeans(n_clusters=3)
    estimator.fit(train_data)

    original_cluster_lists = {
        i: original_test_data.iloc[np.where(estimator.predict(original_test_data.drop('case_id', 1)) == i)[0]]
        for i in range(estimator.n_clusters)}
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    # print original_cluster_lists
    write_header = True
    for cluster_list in cluster_lists:
        clustered_train_data = cluster_lists[cluster_list]

        clustered_test_data = original_cluster_lists[cluster_list]
        original_test_clustered_data = original_cluster_lists[cluster_list]
        clustered_test_data = clustered_test_data.drop('case_id', 1)
        clustered_test_data = clustered_test_data.drop('remaining_time', 1)

        y = clustered_train_data['remaining_time']
        clustered_train_data = clustered_train_data.drop('remaining_time', 1)

        regressor.fit(clustered_train_data, y)

        original_test_clustered_data['prediction'] = regressor.predict(clustered_test_data)

        path = job.get_results_path()
        write_header = write_to_path(original_test_clustered_data, path, write_header)


def no_clustering(original_test_data, train_data, test_data, regressor, job):
    y = train_data['remaining_time']
    print y
    train_data = train_data.drop('remaining_time', 1)

    regressor.fit(train_data, y)

    with open(job.prediction_model_path(), 'wb') as fid:
        cPickle.dump(regressor, fid)

    original_test_data['prediction'] = regressor.predict(test_data)
    original_test_data.to_csv(job.get_results_path(), sep=',', mode='w+', index=False)


def calculate_results(job):
    df = pd.read_csv(filepath_or_buffer=job.get_results_path(), header=0, index_col=0)
    df['remaining_time'] = df['remaining_time'] / 3600
    df['prediction'] = df['prediction'] / 3600
    rmse = sqrt(mean_squared_error(df['remaining_time'], df['prediction']))
    mae = mean_absolute_error(df['remaining_time'], df['prediction'])
    rscore = metrics.r2_score(df['remaining_time'], df['prediction'])

    field_names = ['Run', 'Rmse', 'Mae', 'Rscore']
    row = {'Run': job.method_val(), 'Rmse': rmse, 'Mae': mae, 'Rscore': rscore}
    write_results_to_general(job, field_names, row)
    # DB code. if implemented, remove the above stuff
    row2 = {'uuid': job.uuid, 'rmse': rmse, 'mae': mae, 'rscore': rscore}
    write_results_to_db(row2)

def split_data(data):
    cases = data['case_id'].unique()
    import random
    random.shuffle(cases)

    cases_train_point = int(len(cases) * 0.8)

    train_cases = cases[:cases_train_point]

    ids = []
    for i in range(0, len(data)):
        ids.append(data['case_id'][i] in train_cases)

    train_data = data[ids]
    test_data = data[np.invert(ids)]
    return train_data, test_data


def prep_data(job):
    df = pd.read_csv(filepath_or_buffer=job.get_encoded_file_path(), header=0)
    train_data, test_data = split_data(df)

    train_data = train_data.drop('elapsed_time', 1)
    test_data = test_data.drop('elapsed_time', 1)

    original_test_data = test_data

    train_data = train_data.drop('case_id', 1)
    test_data = test_data.drop('case_id', 1)
    test_data = test_data.drop('remaining_time', 1)

    return train_data, test_data, original_test_data


def __choose_regressor(job):
    regressor = None
    if job.regression == "linear":
        regressor = LinearRegression(fit_intercept=True)
    elif job.regression == "xgboost":
        regressor = xgb.XGBRegressor(n_estimators=2000, max_depth=10)
    elif job.regression == "randomforest":
        regressor = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
    elif job.regression == "lasso":
        regressor = Lasso(fit_intercept=True, warm_start=True)
    return regressor