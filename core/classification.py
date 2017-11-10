import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from core.common import encode, choose_classifier, fast_slow_encode, calculate_results

pd.options.mode.chained_assignment = None


def classifier(job):
    df = encode(job)
    clf = choose_classifier(job)

    df = fast_slow_encode(df, job.rule, job.threshold)

    train_data, test_data, original_test_data = __split_class_data(df)

    if job.clustering == "kmeans":
        results_df, auc = kmeans_clustering(original_test_data, train_data, clf)
    else:
        results_df, auc = no_clustering(original_test_data, train_data, clf)

    results = write_calculate_results(results_df, job, auc)
    return results


def kmeans_clustering(original_test_data, train_data, clf):
    auc = 0
    estimator = KMeans(n_clusters=3)
    estimator.fit(train_data.drop('actual', 1))
    original_cluster_lists = {i: original_test_data.iloc[
        np.where(estimator.predict(original_test_data.drop(['case_id', 'actual'], 1)) == i)[0]] for i in
                              range(estimator.n_clusters)}
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}

    counter = 0
    result_data = None
    for cluster_list in cluster_lists:

        # Train data
        clustered_train_data = cluster_lists[cluster_list]
        y = clustered_train_data['actual']
        # Test data
        original_test_clustered_data = original_cluster_lists[cluster_list]
        actual = original_test_clustered_data['actual']

        if original_test_clustered_data.shape[0] == 0:
            pass
        else:
            clf.fit(clustered_train_data.drop('actual', 1), y)
            prediction = clf.predict(original_test_clustered_data.drop(['case_id', 'actual'], 1))
            scores = clf.predict_proba(original_test_clustered_data.drop(['case_id', 'actual'], 1))

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


def no_clustering(original_test_data, train_data, clf):
    y = train_data['actual']

    clf.fit(train_data.drop('actual', 1), y)

    prediction = clf.predict(original_test_data.drop(['case_id', 'actual'], 1))
    scores = clf.predict_proba(original_test_data.drop(['case_id', 'actual'], 1))[:, 1]
    actual = original_test_data["actual"]
    original_test_data["actual"] = original_test_data["actual"].apply(lambda x: 'Fast' if x else 'Slow')
    original_test_data["predicted"] = prediction
    original_test_data["predicted"] = original_test_data["predicted"].apply(lambda x: 'Fast' if x else 'Slow')

    # FPR,TPR,thresholds_unsorted=
    auc = 0
    try:
        auc = metrics.roc_auc_score(actual, scores)
    except ValueError:
        pass
    return original_test_data, auc


def write_calculate_results(df, job, auc):
    actual_ = df['actual'].values
    predicted_ = df['predicted'].values

    actual_[actual_ == "Fast"] = True
    actual_[actual_ == "Slow"] = False
    predicted_[predicted_ == "Fast"] = True
    predicted_[predicted_ == "Slow"] = False

    f1score, acc = calculate_results(actual_, predicted_)

    row = {'run': job.method_val(), 'f1score': f1score, 'acc': acc, 'auc': auc}
    print("calculation done")
    print(row)
    return row


def __split_class_data(data):
    data = data.sample(frac=1)
    # data = data.drop('elapsed_time', 1)
    data = data.drop('remaining_time', 1)

    # cases_train_point = int(len(data) * 0.8)

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=3)
    original_test_df = test_df
    train_df = train_df.drop('case_id', 1)
    test_df = test_df.drop('case_id', 1)

    return train_df, test_df, original_test_df


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
