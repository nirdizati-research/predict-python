import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from core.common import choose_classifier, calculate_results, fast_slow_encode2, fast_slow_encode
from core.constants import KMEANS, NO_CLUSTER
from django.contrib.admin.templatetags.admin_list import results

pd.options.mode.chained_assignment = None


def classifier(test_df, job, model):
    
    if split['type'] == 'single':
        clf = joblib.load(split['model_path'])
    elif split['type'] =='double':
        clf = joblib.load(split['model_path']) 
        estimator = joblib.load(split['kmean_path'])
        
    test_df = fast_slow_encode(test_df, job['rule'], job['threshold'])

    test_data, original_test_data = drop_columns(test_df)
    if job['clustering'] == KMEANS:
        results_df, auc = kmeans_clustering(original_test_data, clf, estimator)
    elif job['clustering'] == NO_CLUSTER:
        results_df, auc = no_clustering(original_test_data, clf)
    else:
        raise ValueError("Unexpected clustering {}".format(job['clustering']))

    results = prepare_results(results_df, auc)
    return results

def classifier_run(run_df, model):
    split=model['split']

    #run_df = fast_slow_encode(run_df, job['rule'], job['threshold'])
    #run_df = run_df.drop(columns = 'actual')
    
    if split['type'] == 'single':
        clf = joblib.load(split['model_path'])
        result = no_clustering_run(run_df, clf)
    elif split['type'] =='double':
        clf = joblib.load(split['model_path']) 
        estimator = joblib.load(split['kmean_path'])
        result = kmeans_run(run_df, clf, estimator)
    print (result)
    return result

def no_clustering_run(run_df, clf):
    run_df = run_df.drop('trace_id',1)
    results = clf.predict(run_df)
    result = []
    prob = clf.predict_proba(run_df)
    for i in range(len(results)):   
        if results[i]:
            result.append("Fast - prob: {}".format(prob[i]))
        else:
            result.append("Slow - prob: {}".format(prob[i]))
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
        np.where(estimator.predict(original_test_data.drop(['trace_id', 'actual'], 1)) == i)[0]] for i in
                              range(estimator.n_clusters)}
    
    counter = 0
    result_data = None
    for i, cluster_list in original_cluster_lists.items():

        # Test data
        original_test_clustered_data = cluster_list
        actual = original_test_clustered_data['actual']

        if original_test_clustered_data.shape[0] == 0:
            pass
        else:
            prediction = clf[i].predict(original_test_clustered_data.drop(['trace_id', 'actual'], 1))
            scores = clf[i].predict_proba(original_test_clustered_data.drop(['trace_id', 'actual'], 1))

            original_test_clustered_data["predicted"] = prediction
            original_test_clustered_data["predicted"] = original_test_clustered_data["predicted"].map(
                {True: 'Fast', False: 'Slow'})
            original_test_clustered_data["actual"] = original_test_clustered_data["actual"].map(
                {True: 'Fast', False: 'Slow'})

            auc = calculate_auc(actual, scores, auc, counter)
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
    prediction = clf.predict(original_test_data.drop(['trace_id', 'actual'], 1))
    scores = clf.predict_proba(original_test_data.drop(['trace_id', 'actual'], 1))[:, 1]
    actual = original_test_data["actual"]
    original_test_data["actual"] = original_test_data["actual"].map(
        {True: 'Fast', False: 'Slow'})
    original_test_data["predicted"] = prediction
    original_test_data["predicted"] = original_test_data["predicted"].map(
        {True: 'Fast', False: 'Slow'})

    # FPR,TPR,thresholds_unsorted=
    auc = 0
    try:
        auc = metrics.roc_auc_score(actual, scores)
    except ValueError:
        pass
    return original_test_data, auc


def prepare_results(df, auc: int):
    actual_ = df['actual'].values
    predicted_ = df['predicted'].values

    actual_[actual_ == "Fast"] = True
    actual_[actual_ == "Slow"] = False
    predicted_[predicted_ == "Fast"] = True
    predicted_[predicted_ == "Slow"] = False

    f1score, acc = calculate_results(actual_, predicted_)

    row = {'f1score': f1score, 'acc': acc, 'auc': auc}
    return row


def drop_columns(test_df):
    
    # original_test_df = test_df
    original_test_df = test_df.drop('remaining_time', 1)
    test_df = test_df.drop(['remaining_time', 'trace_id'], 1)
    return test_df, original_test_df


def calculate_auc(actual, scores, auc: int, counter: int):
    if scores.shape[1] == 1:
        auc += 0
    else:
        try:
            auc += metrics.roc_auc_score(actual, scores[:, 1])
            counter += 1
        except Exception:
            auc += 0
    return auc
