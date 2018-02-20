import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from core.common import choose_classifier, calculate_results
from core.constants import KMEANS

pd.options.mode.chained_assignment = None


def next_activity(training_df, test_df, job):
    clf = choose_classifier(job['method'])

    train_data, test_data, original_test_data = drop_columns(training_df, test_df)

    if job['clustering'] == KMEANS:
        results_df, auc = kmeans_clustering(original_test_data, train_data, clf)
    else:
        results_df, auc = no_clustering(original_test_data, train_data, clf)

    results = prepare_results(results_df, auc)
    return results

def next_activity_run(run_df, model, job):
    split = model['split']
    run_df = run_df.drop('label', 1)
    if split['type'] == 'single':
        clf = joblib.load(split['model_path'])
        result = no_clustering_run(run_df,clf)
    elif split[type] == 'double':
        clf = joblib.load(split['model_path'])
        estimator = joblib.load(split['kmean_path'])
        result = kmeans_run(run_df, clf, estimator)
    
    results = clf.predict(run_df)
    prob = clf.predict_proba(run_df)
    result= "{} - prob {}".format(results, prob)
    print(results)
    return result

def no_clustering_run(run_df, clf):
    run_df = run_df.drop('trace_id',1)
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

def kmeans_clustering(original_test_data, train_data, clf):
    auc = 0
    estimator = KMeans(n_clusters=3)
    estimator.fit(train_data.drop('label', 1))
    original_cluster_lists = {i: original_test_data.iloc[
        np.where(estimator.predict(original_test_data.drop(['trace_id', 'label'], 1)) == i)[0]] for i in
                              range(estimator.n_clusters)}
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}

    counter = 0
    result_data = None
    for cluster_list in cluster_lists:

        # Train data
        clustered_train_data = cluster_lists[cluster_list]
        y = clustered_train_data['label']
        # Test data
        original_test_clustered_data = original_cluster_lists[cluster_list]
        actual = original_test_clustered_data['label']

        if original_test_clustered_data.shape[0] == 0:
            pass
        else:
            clf.fit(clustered_train_data.drop('label', 1), y)
            prediction = clf.predict(original_test_clustered_data.drop(['trace_id', 'label'], 1))

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


def no_clustering(original_test_data, train_data, clf):
    y = train_data['label']
    train_data = train_data.drop('label', 1)
    clf.fit(train_data, y)

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

    f1score, acc = calculate_results(actual_, predicted_)

    row = {'f1score': f1score, 'acc': acc, 'auc': auc}
    return row


def drop_columns(training_df, test_df):
    original_test_df = test_df
    train_df = training_df.drop('trace_id', 1)
    test_df = test_df.drop('trace_id', 1)

    return train_df, test_df, original_test_df
