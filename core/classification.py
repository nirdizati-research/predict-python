from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core.common import get_method_config
from core.constants import KNN, RANDOM_FOREST, DECISION_TREE, XGBOOST
from utils.result_metrics import calculate_results_binary_classification, calculate_results_multiclass_classification, \
    calculate_auc
from encoders.common import REMAINING_TIME, ATTRIBUTE_NUMBER, ATTRIBUTE_STRING, NEXT_ACTIVITY, DURATION
from core.constants import KMEANS, NO_CLUSTER

pd.options.mode.chained_assignment = None


def classification(training_df: DataFrame, test_df: DataFrame, job: dict):
    is_binary_classifier = _check_is_binary_classifier(job['label'].type)

    classifier = _choose_classifier(job)

    train_data, test_data, original_test_data = _drop_columns(training_df, test_df)

    if job['clustering'] == KMEANS:
        results_df, auc, model_split = _kmeans_clustering_train(original_test_data, train_data, classifier,
                                                                job['kmeans'], is_binary_classifier)
    elif job['clustering'] == NO_CLUSTER:
        results_df, auc, model_split = _no_clustering_train(original_test_data, train_data, classifier,
                                                            is_binary_classifier)
    else:
        raise ValueError("Unexpected clustering {}".format(job['clustering']))

    results = _prepare_results(results_df, auc, is_binary_classifier)
    return results, model_split


def classification_single_log(run_df: DataFrame, model: dict):
    result = None
    is_binary_classifier = _check_is_binary_classifier(model['label'].type)

    split = model['split']
    results = dict()
    results['label'] = run_df['label']
    if is_binary_classifier:
        run_df = run_df.drop('label', 1)

    if split['type'] == NO_CLUSTER:
        clf = joblib.load(split['model_path'])
        result, _ = _no_clustering_test(run_df, clf)
    elif split['type'] == KMEANS:
        clf = joblib.load(split['model_path'])
        estimator = joblib.load(split['estimator_path'])
        result, _ = _kmeans_clustering_test(run_df, clf, estimator, is_binary_classifier)
    results['prediction'] = result['predicted']
    return results


def _kmeans_clustering_train(original_test_data, train_data, classifier, kmeans_config: dict, is_binary_classifier: bool):
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
    result, auc = _kmeans_clustering_test(original_test_data, models, estimator, is_binary_classifier, testing=True)
    return result, auc, model_split


def _kmeans_clustering_test(test_data, classifier, estimator, is_binary_classifier: bool, testing: bool = False):
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


def _no_clustering_train(original_test_data, train_data, classifier: Any, is_binary_classifier: bool):
    y = train_data['label']
    try:
        classifier.fit(train_data.drop('label', 1), y)
    except:
        classifier.partial_fit(train_data.drop('label', 1).values, y)

    actual = original_test_data['label']
    original_test_data, scores = _no_clustering_test(original_test_data, classifier, True)

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


def _no_clustering_test(test_data, classifier, testing=False):
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


def _prepare_results(df: DataFrame, auc: int, is_binary_classifier: bool):
    actual = df['label'].values
    predicted = df['predicted'].values

    if is_binary_classifier:
        row = calculate_results_binary_classification(actual, predicted)
    else:
        row = calculate_results_multiclass_classification(actual, predicted)
    row['auc'] = auc
    return row


def _drop_columns(train_df: DataFrame, test_df: DataFrame):
    original_test_df = test_df
    train_df = train_df.drop('trace_id', 1)
    test_df = test_df.drop('trace_id', 1)
    return train_df, test_df, original_test_df


def _choose_classifier(job: dict):
    method, config = get_method_config(job)
    print("Using method {} with config {}".format(method, config))
    if method == KNN:
        clf = KNeighborsClassifier(**config)
    elif method == RANDOM_FOREST:
        clf = RandomForestClassifier(**config)
    elif method == DECISION_TREE:
        clf = DecisionTreeClassifier(**config)
    elif method == XGBOOST:
        clf = XGBClassifier(**config)
    else:
        raise ValueError("Unexpected classification method {}".format(method))
    return clf

def _check_is_binary_classifier(label_type):
    if label_type in [REMAINING_TIME, ATTRIBUTE_NUMBER, DURATION]:
        return True
    elif label_type in [NEXT_ACTIVITY, ATTRIBUTE_STRING]:
        return False
    else:
        raise ValueError("Label type not supported", label_type)
