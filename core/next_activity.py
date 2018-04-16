import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from core.common import choose_classifier, calculate_results
from core.constants import KMEANS

pd.options.mode.chained_assignment = None


def next_activity(training_df, test_df, job: dict):
    clf = choose_classifier(job)

    train_data, test_data, original_test_data = drop_columns(training_df, test_df)

    if job['clustering'] == KMEANS:
        results_df, auc, split = kmeans_clustering_train(original_test_data, train_data, clf)
    else:
        results_df, auc, split = no_clustering_train(original_test_data, train_data, clf)

    results = prepare_results(results_df, auc)
    return results, split


def kmeans_clustering_train(original_test_data, train_data, clf):
    estimator = KMeans(n_clusters=3)
    models = dict()
    estimator.fit(train_data.drop('label', 1))
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    for i, cluster_list in cluster_lists.items():
        clustered_train_data = cluster_list
        if clustered_train_data.shape[0] == 0:
            pass
        else:
            y = clustered_train_data['label']
            clf.fit(clustered_train_data.drop('label', 1), y)

            models[i] = clf
    split=dict()
    split['type']='double'
    split['estimator']=estimator
    split['model']=models
    result, auc = kmeans_clustering_test(original_test_data, models, estimator, testing=True)
    return result, auc, split


def kmeans_clustering_test(test_data, clf, estimator, testing=False):
    drop_list = ['trace_id', 'label'] if testing else ['trace_id']
    auc = 0
    counter = 0
    test_cluster_lists = {
        i: test_data.iloc[np.where(estimator.predict(test_data.drop(drop_list, 1)) == i)[0]]
        for i in range(estimator.n_clusters)}
    result_data = None
    for i, cluster_list in test_cluster_lists.items():
        original_clustered_test_data = cluster_list
        if original_clustered_test_data.shape[0] == 0:
            pass
        else:
            clustered_test_data = original_clustered_test_data.drop(drop_list, 1)

            prediction = clf[i].predict(clustered_test_data)

            original_clustered_test_data["predicted"] = prediction
            if testing:
                original_clustered_test_data["actual"] = original_clustered_test_data['label']

            if result_data is None:
                result_data = original_clustered_test_data
            else:
                result_data = result_data.append(original_clustered_test_data)
    if testing:
        try:
            auc = float(auc) / counter
        except ZeroDivisionError:
            auc = 0
    return result_data, auc


def no_clustering_train(original_test_data, train_data, clf):
    y = train_data['label']

    clf.fit(train_data.drop('label', 1), y)
    actual = original_test_data["label"]
    original_test_data, scores = no_clustering_test(original_test_data.drop('label', 1), clf)
    original_test_data["actual"] = actual
    # TODO calculate AUC
    auc = 0
    split=dict()
    split['type']='single'
    split['model']=clf
    return original_test_data, auc, split


def no_clustering_test(test_data, clf):
    prediction = clf.predict(test_data.drop('trace_id', 1))
    scores = clf.predict_proba(test_data.drop('trace_id', 1))[:, 1]
    test_data["predicted"] = prediction
    return test_data, scores


def prepare_results(df, auc):
    actual_ = df['actual'].values
    predicted_ = df['predicted'].values

    row = calculate_results(actual_, predicted_)
    row['auc'] = auc
    return row


def drop_columns(training_df, test_df):
    original_test_df = test_df
    train_df = training_df.drop('trace_id', 1)
    test_df = test_df.drop('trace_id', 1)

    return train_df, test_df, original_test_df
