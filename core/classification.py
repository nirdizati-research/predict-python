import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from core.common import choose_classifier, calculate_results
from core.constants import KMEANS, NO_CLUSTER

pd.options.mode.chained_assignment = None


def classification(training_df, test_df, job, is_binary_classifier):
    classifier = choose_classifier(job)

    train_data, test_data, original_test_data = drop_columns(training_df, test_df)

    if job['clustering'] == KMEANS:
        results_df, auc, model_split = kmeans_clustering_train(original_test_data, train_data, classifier,
                                                               job['kmeans'], is_binary_classifier)
    elif job['clustering'] == NO_CLUSTER:
        results_df, auc, model_split = no_clustering_train(original_test_data, train_data, classifier,
                                                           is_binary_classifier)
    else:
        raise ValueError("Unexpected clustering {}".format(job['clustering']))

    results = prepare_results(results_df, auc, is_binary_classifier)
    return results, model_split


def classification_single_log(run_df, model, is_binary_classifier):
    result = None

    split = model['split']
    results = dict()
    results['label'] = run_df['label']
    if is_binary_classifier:
        run_df = run_df.drop('label', 1)

    if split['type'] == NO_CLUSTER:
        clf = joblib.load(split['model_path'])
        result, _ = no_clustering_test(run_df, clf)
    elif split['type'] == KMEANS:
        clf = joblib.load(split['model_path'])
        estimator = joblib.load(split['estimator_path'])
        result, _ = kmeans_clustering_test(run_df, clf, estimator, is_binary_classifier)
    results['prediction'] = result['predicted']
    return results


def kmeans_clustering_train(original_test_data, train_data, classifier, kmeans_config: dict, is_binary_classifier):
    estimator = KMeans(**kmeans_config)
    models = dict()

    estimator.fit(train_data.drop('label', 1))
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    for i, cluster_list in cluster_lists.items():
        clustered_train_data = cluster_list
        if clustered_train_data.shape[0] == 0:
            pass
        else:
            y = clustered_train_data['label']
            classifier.fit(clustered_train_data.drop('label', 1), y)
            models[i] = classifier
    model_split = dict()
    model_split['type'] = KMEANS
    model_split['estimator'] = estimator
    model_split['model'] = models
    result, auc = kmeans_clustering_test(original_test_data, models, estimator, is_binary_classifier, testing=True)
    return result, auc, model_split


def kmeans_clustering_test(test_data, classifier, estimator, is_binary_classifier, testing=False):
    drop_list = ['trace_id', 'label'] if testing else ['trace_id']
    auc = 0
    counter = 0

    test_cluster_lists = {
        i: test_data.iloc[np.where(estimator.predict(test_data.drop(drop_list, 1)) == i)[0]]
        for i in range(estimator.n_clusters)}
    result_data = None
    for i, cluster_list in test_cluster_lists.items():
        counter += 1
        original_clustered_test_data = cluster_list
        if original_clustered_test_data.shape[0] == 0:
            pass
        else:
            clustered_test_data = original_clustered_test_data.drop(drop_list, 1)

            prediction = classifier[i].predict(clustered_test_data)
            original_clustered_test_data["predicted"] = prediction

            if is_binary_classifier:
                scores = classifier[i].predict_proba(clustered_test_data)
                if testing:
                    actual = original_clustered_test_data['label']
                    auc = calculate_auc(actual, scores, auc)
            else:
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


def no_clustering_train(original_test_data, train_data, classifier, is_binary_classifier):
    y = train_data['label']
    try:
        classifier.fit(train_data.drop('label', 1), y)
    except:
        classifier.partial_fit(train_data.drop('label', 1).values, y)

    actual = original_test_data['label']
    original_test_data, scores = no_clustering_test(original_test_data, classifier, True)

    auc = 0
    if is_binary_classifier:
        try:
            auc = metrics.roc_auc_score(actual, scores)
        except ValueError:
            pass
    else:
        pass  # TODO: check if AUC is ok for multiclass, otherwise implement
    model_split = dict()
    model_split['type'] = NO_CLUSTER
    model_split['model'] = classifier
    return original_test_data, auc, model_split


def no_clustering_test(test_data, classifier, testing=False):
    scores = 0
    if testing:
        if hasattr(classifier, 'decision_function'):
            scores = classifier.decision_function(test_data.drop(['trace_id', 'label'], 1))
        else:
            scores = classifier.predict_proba(test_data.drop(['trace_id', 'label'], 1))
            if np.size(scores, 1) >= 2: # checks number of columns
                scores = scores[:, 1]
    test_data['predicted'] = classifier.predict(test_data.drop(['trace_id', 'label'], 1))
    return test_data, scores


def prepare_results(df, auc: int, is_binary_classifier):
    actual = df['label'].values
    predicted = df['predicted'].values

    if is_binary_classifier:
        row = calculate_results(actual, predicted)
    else:
        row = results_multiclass_label(actual, predicted)
    row['auc'] = auc
    return row


def drop_columns(train_df, test_df):
    original_test_df = test_df
    train_df = train_df.drop('trace_id', 1)
    test_df = test_df.drop('trace_id', 1)
    return train_df, test_df, original_test_df


def calculate_auc(actual, scores, auc: int):
    if scores.shape[1] == 1:
        auc += 0
    else:
        try:
            auc += metrics.roc_auc_score(actual, scores[:, 1])
        except Exception:
            pass
    return auc


def results_multiclass_label(actual: list, predicted: list):
    # average is needed as these are multi-label lists
    # print(classification_report(actual, predicted))
    acc = accuracy_score(actual, predicted)
    f1score = f1_score(actual, predicted, average='macro')
    precision = precision_score(actual, predicted, average='macro')
    recall = recall_score(actual, predicted, average='macro')

    if len(set(actual + predicted)) == 2:
        row = calculate_results([el == 'true' for el in actual], [el == 'true' for el in predicted])
    else:
        row = {'f1score': f1score, 'acc': acc, 'precision': precision, 'recall': recall}
    return row
