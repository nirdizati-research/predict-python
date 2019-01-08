import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from skmultiflow.trees import HoeffdingTree, HAT

from core.common import choose_classifier, calculate_results, add_actual
from core.constants import KMEANS, NO_CLUSTER, HOEFFDING_TREE, ADAPTIVE_TREE

pd.options.mode.chained_assignment = None


def binary_classifier(training_df, test_df, job):
    """For True/False classification"""
    clf = choose_classifier(job)

    training_df, test_df = add_actual(training_df, test_df)

    train_data, test_data, original_test_data = drop_columns(training_df, test_df)
    if job['clustering'] == KMEANS:
        results_df, auc, model_split = kmeans_clustering_train(original_test_data, train_data, clf, job['kmeans'])
    elif job['clustering'] == NO_CLUSTER:
        results_df, auc, model_split = no_clustering_train(original_test_data, train_data, clf)
    else:
        raise ValueError("Unexpected clustering {}".format(job['clustering']))

    results = prepare_results(results_df, auc)
    return results, model_split


def binary_classifier_single_log(run_df, model):
    split=model['split']
    results = dict()
    results['label'] = run_df['label']
    run_df = run_df.drop('label', 1)
    if split['type'] == NO_CLUSTER:
        clf = joblib.load(split['model_path'])
        result, _ = no_clustering_test(run_df, clf)
    elif split['type'] ==KMEANS:
        clf = joblib.load(split['model_path']) 
        estimator = joblib.load(split['estimator_path'])
        result, _ = kmeans_test(run_df, clf, estimator)
    results['prediction'] = result['predicted']
    return results


def kmeans_clustering_train(original_test_data, train_data, clf, kmeans_config: dict):
    estimator = KMeans(**kmeans_config)
    models = dict()
    estimator.fit(train_data.drop('actual', 1))
    cluster_lists = {i: train_data.iloc[np.where(estimator.labels_ == i)[0]] for i in range(estimator.n_clusters)}
    for i, cluster_list in cluster_lists.items():
        clustered_train_data = cluster_list
        if clustered_train_data.shape[0] == 0:
            pass
        else:
            y = clustered_train_data['actual']
            clf.fit(clustered_train_data.drop('actual', 1), y)
            models[i] = clf
    model_split = dict()
    model_split['type'] = KMEANS
    model_split['estimator'] = estimator
    model_split['model'] = models
    result, auc = kmeans_clustering_test(original_test_data, models, estimator, testing=True)
    return result, auc, model_split


def kmeans_clustering_test(test_data, clf, estimator, testing=False):
    drop_list = ['trace_id', 'actual'] if testing else ['trace_id']
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

            prediction = clf[i].predict(clustered_test_data)
            scores = clf[i].predict_proba(clustered_test_data)

            original_clustered_test_data["predicted"] = prediction
            if testing:
                actual = original_clustered_test_data['actual']
                auc = calculate_auc(actual, scores, auc)

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
    y = train_data['actual']
    # if isinstance(clf, MultinomialNB):
    #     clf.__init__(alpha=clf.get_params()[ 'alpha' ], class_prior=list( y.value_counts(normalize=True) ),
    #                  fit_prior=True)
    try:
        clf.fit(train_data.drop('actual', 1), y)
    except:
        clf.partial_fit(train_data.drop('actual', 1).values, y)
    actual = original_test_data["actual"]
    original_test_data, scores = no_clustering_test(original_test_data.drop('actual', 1), clf, True)
    original_test_data["actual"] = actual

    if isinstance(clf, HoeffdingTree) or isinstance(clf, HAT):
        print()
        print('\tRESULTING TREE:')
        print(clf.get_model_description())
        print()
    elif isinstance(clf, MultinomialNB):
        print()
        print('\tRESULTING PRIOR:')
        print(np.exp(clf.class_log_prior_))
        print()

    auc = 0
    try:
        auc = metrics.roc_auc_score(actual, scores)
    except ValueError:
        print('ValueError in AUC_ROC')
        pass
    model_split = dict()
    model_split['type'] = NO_CLUSTER
    model_split['model'] = clf
    return original_test_data, auc, model_split


def no_clustering_update(original_test_data, train_data, clf):
    y = train_data['actual']

    if clf.__class__.__name__ == HOEFFDING_TREE or clf.__class__.__name__ == ADAPTIVE_TREE:
        _train_data = train_data.drop('actual', 1).values
    else:
        _train_data = train_data.drop('actual', 1)

    clf.partial_fit(_train_data, y)

    actual = original_test_data["actual"]

    original_test_data, scores = no_clustering_test(original_test_data.drop('actual', 1), clf, True)

    original_test_data["actual"] = actual

    auc = 0
    try:
        auc = metrics.roc_auc_score(actual, scores)
    except ValueError:
        pass
    model_split = dict()
    model_split[ 'type' ] = NO_CLUSTER
    model_split[ 'model' ] = clf
    return original_test_data, auc, model_split, clf


def no_clustering_test(test_data, clf, testing=False):
    if clf.__class__.__name__ == HOEFFDING_TREE or clf.__class__.__name__ == ADAPTIVE_TREE:
        _test_data = test_data.drop('trace_id', 1).values
    else:
        _test_data = test_data.drop('trace_id', 1)
    prediction = clf.predict(_test_data)
    scores = 0
    if testing:
        try:
            scores = clf.decision_function(_test_data)
        except:
            scores = clf.predict_proba(_test_data)[:, 1]
    test_data["predicted"] = prediction
    return test_data, scores


def prepare_results(df, auc: int):
    actual_ = df['actual'].values
    predicted_ = df['predicted'].values

    row = calculate_results(actual_, predicted_)
    row['auc'] = auc
    return row


def drop_columns(training_df, test_df):
    training_df = training_df.drop(['label', 'trace_id'], 1)
    original_test_df = test_df.drop('label', 1)
    test_df = test_df.drop(['label', 'trace_id'], 1)
    return training_df, test_df, original_test_df


def calculate_auc(actual, scores, auc: int):
    if scores.shape[1] == 1:
        auc += 0
    else:
        try:
            auc += metrics.roc_auc_score(actual, scores[:, 1])
        except Exception:
            pass
    return auc
