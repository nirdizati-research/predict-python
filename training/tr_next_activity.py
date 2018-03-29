import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from core.common import choose_classifier
from core.constants import KMEANS

pd.options.mode.chained_assignment = None


def tr_next_activity(training_df, job):

    train_data= drop_columns(training_df)
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
    estimator.fit(train_data.drop('label', 1))
    models=dict()
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    for i, cluster_list in cluster_lists.items():
        clustered_train_data = cluster_lists[i]
        if clustered_train_data.shape[0] == 0:
            pass
        else:
        # Train data
            clf = choose_classifier(job['method'])
            y = clustered_train_data['label']

            clf.fit(clustered_train_data.drop('label', 1), y)


            models[i] = clf
    return models, estimator


def no_clustering(train_data, job):
    clf = choose_classifier(job['method'])
    y = train_data['label']
    train_data = train_data.drop('label', 1)
    clf.fit(train_data, y)

    return clf

def drop_columns(training_df):
    train_df = training_df.drop('trace_id', 1)

    return train_df
