import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from core.common import choose_classifier, calculate_results
from core.constants import KMEANS

pd.options.mode.chained_assignment = None


def next_activity(test_df, job, model):
    if split['type'] == 'single':
        clf = joblib.load(split['model_path'])
    elif split[type] == 'double':
        clf = joblib.load(split['model_path'])
        estimator = joblib.load(split['kmean_path'])

    test_data, original_test_data = drop_columns(test_df)

    if job['clustering'] == KMEANS:
        results_df, auc = kmeans_clustering(original_test_data, clf, estimator)
    else:
        results_df, auc = no_clustering(original_test_data, clf)

    results = prepare_results(results_df, auc)
    return results

def next_activity_run(run_df, model):
    split = model['split']
    if split['type'] == 'single':
        clf = joblib.load(split['model_path'])
        result = no_clustering_run(run_df,clf)
    elif split[type] == 'double':
        clf = joblib.load(split['model_path'])
        estimator = joblib.load(split['kmean_path'])
        result = kmeans_run(run_df, clf, estimator)
    return result

def no_clustering_run(run_df, clf):
    run_df = run_df.drop('trace_id',1)
    print(run_df)
    results = clf.predict(run_df)
    result = []
    prob = clf.predict_proba(run_df)
    for i in range(len(results)):   
        if results[i]:
            result.append("{} - prob: {}".format(results[i],prob[i]))
        else:
            result.append("{} - prob: {}".format(results[i],prob[i]))
    return result

def kmeans_run(run_df, clf, estimator):
    test_cluster_lists = {
        i: run_df.iloc[np.where(estimator.predict(run_df.drop('trace_id', 1)) == i)[0]]
        for i in range(estimator.n_clusters)}
    results = None
    for i,cluster_list in test_cluster_lists.items():
        clustered_test_data = test_cluster_lists[i]
        if clustered_test_data.shape[0] == 0:
            pass
        else:
            clustered_test_data['result']=clf[i].predict(clustered_test_data)
            
    return clustered_test_data['result']

def kmeans_clustering(original_test_data, clf, estimator):
    auc = 0
    original_cluster_lists = {i: original_test_data.iloc[
        np.where(estimator.predict(original_test_data.drop(['trace_id', 'label'], 1)) == i)[0]] for i in
                              range(estimator.n_clusters)}

    counter = 0
    result_data = None
    for i,cluster_list in original_cluster_lists.items():

        # Test data
        original_test_clustered_data = cluster_list
        actual = original_test_clustered_data['label']

        if original_test_clustered_data.shape[0] == 0:
            pass
        else:
            prediction = clf[i].predict(original_test_clustered_data.drop(['trace_id', 'label'], 1))

            original_test_clustered_data["predicted"] = prediction
            original_test_clustered_data["actual"] = actual

            if result_data is None:
                result_data = original_test_clustered_data
            else:
                result_data = result_data.append(original_test_clustered_data)
    try:
        auc = float(auc) / counter
    except ZeroDivisionError:
        auc = 0
    return result_data, auc


def no_clustering(original_test_data, clf):
    
    actual = original_test_data["label"]
    original_test_data = original_test_data.drop(['trace_id', 'label'], 1)
    prediction = clf.predict(original_test_data)
    # scores = clf.predict_proba(original_test_data)[:, 1]

    original_test_data["actual"] = actual
    original_test_data["predicted"] = prediction

    # TODO calculate AUC
    auc = 0
    return original_test_data, auc


def prepare_results(df, auc):
    actual_ = df['actual'].values
    predicted_ = df['predicted'].values

    row = calculate_results(actual_, predicted_)
    row['auc'] = auc
    return row


def drop_columns(test_df):
    original_test_df = test_df
    test_df = test_df.drop('trace_id', 1)

    return test_df, original_test_df
