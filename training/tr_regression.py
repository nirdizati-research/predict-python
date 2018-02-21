import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .models import Split

from core.constants import KMEANS, LINEAR, RANDOM_FOREST, LASSO, NO_CLUSTER

pd.options.mode.chained_assignment = None


def tr_regression(training_df, job):
    train_data = prep_data(training_df)
    split = dict()
    if job['clustering'] == KMEANS:
        model, estimator = kmeans_clustering(train_data, job)
        split['type']='double'
        split['model']=model
        split['estimator']=estimator
    else:
        model = no_clustering(train_data, job)
        split['type']='single'
        split['model']=model
    return split


def kmeans_clustering(train_data, job):
    estimator = KMeans(n_clusters=3)
    estimator.fit(train_data.drop('remaining_time',1))
    models = dict()
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    #print(cluster_lists)
    for i, cluster_list in cluster_lists.items():
        clustered_train_data = cluster_lists[i]
        if clustered_train_data.shape[0] == 0:
            pass
        else:
            regressor = __choose_regressor(job['method'])
            y = clustered_train_data['remaining_time']
            clustered_train_data = clustered_train_data.drop('remaining_time', 1)

            regressor.fit(clustered_train_data, y)
            models[i] = regressor
    return models, estimator

def no_clustering(train_data, job):
    regressor = __choose_regressor(job['method'])
    y = train_data['remaining_time']
    train_data = train_data.drop('remaining_time', 1)
    regressor.fit(train_data, y)
    return regressor

def prep_data(training_df):
    train_data = training_df.drop(['elapsed_time', 'trace_id'], 1)
    

    return train_data

def __choose_regressor(regression_type: str):
    regressor = None
    if regression_type == LINEAR:
        regressor = LinearRegression(fit_intercept=True)
    elif regression_type == RANDOM_FOREST:
        regressor = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
    elif regression_type == LASSO:
        regressor = Lasso(fit_intercept=True, warm_start=True)
    return regressor
