import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans

from core.common import choose_classifier, calculate_results, fast_slow_encode
from core.constants import KMEANS, NO_CLUSTER

pd.options.mode.chained_assignment = None


def tr_classifier(training_df, job):
    clf = choose_classifier(job['method'])

    training_df= fast_slow_encode(training_df, job['rule'], job['threshold'])

    train_data = drop_columns(training_df)
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
    models = dict()
    estimator.fit(train_data.drop('actual', 1))
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    for i, cluster_list in cluster_lists.items():
        clustered_train_data = cluster_lists[i]
        if clustered_train_data.shape[0] == 0:
            pass
        else:
        # Train data
            clf = choose_classifier(job['method'])
            y = clustered_train_data['actual']
        
            clf.fit(clustered_train_data.drop('actual', 1), y)
        
            models[i] = clf
    return models, estimator


def no_clustering(train_data, job):
    clf = choose_classifier(job['method'])
    y = train_data['actual']

    clf.fit(train_data.drop('actual', 1), y)
    
    return clf

def drop_columns(training_df):
    training_df = training_df.drop(['remaining_time', 'trace_id'], 1)
    return training_df

