from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import clone
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.trees import HoeffdingTree, HAT
from xgboost import XGBClassifier

from src.clustering.clustering import Clustering
from src.core.common import get_method_config
from src.encoding.models import Encoding
from src.jobs.models import Job, ModelType
from src.labelling.models import LabelTypes
from src.predictive_model.classification.custom_classification_models import NNClassifier
from src.predictive_model.classification.models import ClassificationMethods
from src.utils.django_orm import duplicate_orm_row
from src.utils.result_metrics import calculate_results_classification, get_auc

pd.options.mode.chained_assignment = None

import logging

logger = logging.getLogger(__name__)


def classification(training_df: DataFrame, test_df: DataFrame, clusterer: Clustering, job: Job) -> (dict, dict):
    """main classification entry point

    train and tests the classifier using the provided data

    :param clusterer:
    :param training_df: training DataFrame
    :param test_df: testing DataFrame
    :param job: job configuration
    :return: predictive_model scores and split

    """
    train_data = _drop_columns(training_df)
    test_data = _drop_columns(test_df)

    # job.encoding = duplicate_orm_row(Encoding.objects.filter(pk=job.encoding.pk)[0])  # TODO: maybe here would be better an intelligent get_or_create...
    job.encoding = Encoding.objects.create(
        data_encoding=job.encoding.data_encoding,
        value_encoding=job.encoding.value_encoding,
        add_elapsed_time=job.encoding.add_elapsed_time,
        add_remaining_time=job.encoding.add_remaining_time,
        add_executed_events=job.encoding.add_executed_events,
        add_resources_used=job.encoding.add_resources_used,
        add_new_traces=job.encoding.add_new_traces,
        features=job.encoding.features,
        prefix_length=job.encoding.prefix_length,
        padding=job.encoding.padding,
        task_generation_type=job.encoding.task_generation_type
    )
    job.encoding.features = list(train_data.columns.values)
    job.encoding.save()
    job.save()

    model_split = _train(train_data, _choose_classifier(job), clusterer)
    results_df, auc = _test(
        model_split,
        test_data,
        evaluation=True,
        is_binary_classifier=_check_is_binary_classifier(job.labelling.type)
    )

    results = _prepare_results(results_df, auc)

    return results, model_split


def update_and_test(training_df: DataFrame, test_df: DataFrame, job: Job):
    """

    :param training_df:
    :param test_df:
    :param job:
    :return:
    """
    train_data = _drop_columns(training_df)
    test_data = _drop_columns(test_df)

    job.encoding = job.incremental_train.encoding
    job.encoding.save()
    job.save()

    if list(train_data.columns.values) != job.incremental_train.encoding.features:
        # TODO: how do I align the two feature vectors?
        train_data, _ = train_data.align(
            pd.DataFrame(columns=job.incremental_train.encoding.features), axis=1, join='right')
        train_data = train_data.fillna(0)
        test_data, _ = test_data.align(
            pd.DataFrame(columns=job.incremental_train.encoding.features), axis=1, join='right')
        test_data = test_data.fillna(0)

    # TODO: UPDATE if incremental, otherwise just test
    model_split = _update(job, train_data)

    results_df, auc = _test(model_split, test_data, evaluation=True,
                            is_binary_classifier=_check_is_binary_classifier(job.labelling.type))

    results = _prepare_results(results_df, auc)

    return results, model_split


def _train(train_data: DataFrame, classifier: ClassifierMixin, clusterer: Clustering) -> dict:
    """Initializes and train the predictive model with the given data

    :param train_data:
    :param classifier:
    :param clusterer:
    :return:
    """
    models = dict()

    train_data = clusterer.cluster_data(train_data)

    for cluster in range(clusterer.n_clusters):
        cluster_train_df = train_data[cluster]
        if not cluster_train_df.empty:
            cluster_targets_df = DataFrame(cluster_train_df['label'])
            try:
                classifier.fit(cluster_train_df.drop('label', 1), cluster_targets_df.values.ravel())
            except (NotImplementedError, KeyError):
                classifier.partial_fit(cluster_train_df.drop('label', 1).values, cluster_targets_df.values.ravel())
            except Exception as exception:
                raise exception

            models[cluster] = classifier
            try:
                classifier = clone(classifier)
            except TypeError:
                classifier = clone(classifier, safe=False)
                classifier.reset()

    return {ModelType.CLUSTERER.value: clusterer, ModelType.CLASSIFIER.value: models}


def _update(job: Job, data: DataFrame) -> dict:
    """Updates the existing model

    :param job:
    :param data:
    :return:
    """
    previous_job = job.incremental_train

    clusterer = Clustering.load_model(previous_job)

    update_data = clusterer.cluster_data(data)

    models = joblib.load(previous_job.predictive_model.model_path)
    if job.predictive_model.prediction_method in [ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value,
                                                  ClassificationMethods.ADAPTIVE_TREE.value,
                                                  ClassificationMethods.HOEFFDING_TREE.value,
                                                  ClassificationMethods.SGDCLASSIFIER.value,
                                                  ClassificationMethods.PERCEPTRON.value,
                                                  ClassificationMethods.RANDOM_FOREST.value]:  # TODO: workaround
        print('entered update')
        for cluster in range(clusterer.n_clusters):
            x = update_data[cluster]
            if not x.empty:
                y = x['label']
                try:
                    if previous_job.predictive_model.prediction_method == ClassificationMethods.RANDOM_FOREST.value:
                        models[cluster].fit(x.drop('label', 1), y.values.ravel())
                    else:
                        models[cluster].partial_fit(x.drop('label', 1), y.values.ravel())
                except (NotImplementedError, KeyError):
                    if previous_job.predictive_model.prediction_method == ClassificationMethods.RANDOM_FOREST.value:
                        models[cluster].fit(x.drop('label', 1).values, y.values.ravel())
                    else:
                        models[cluster].partial_fit(x.drop('label', 1).values, y.values.ravel())
                except Exception as exception:
                    raise exception

    return {ModelType.CLUSTERER.value: clusterer, ModelType.CLASSIFIER.value: models}


def _test(model_split: dict, test_data: DataFrame, evaluation: bool, is_binary_classifier: bool) -> (DataFrame, float):
    """Tests the given predictive model with the given data

    :param model_split:
    :param test_data:
    :param evaluation:
    :param is_binary_classifier:
    :return:
    """
    clusterer = model_split[ModelType.CLUSTERER.value]
    classifier = model_split[ModelType.CLASSIFIER.value]

    test_data = clusterer.cluster_data(test_data)

    results_df = DataFrame()
    auc = 0

    non_empty_clusters = clusterer.n_clusters

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = test_data[cluster]
        if cluster_test_df.empty:
            non_empty_clusters -= 1
        else:
            cluster_targets_df = cluster_test_df['label']
            if evaluation:
                try:
                    if hasattr(classifier[cluster], 'decision_function'):
                        scores = classifier[cluster].decision_function(cluster_test_df.drop(['label'], 1))
                    else:
                        scores = classifier[cluster].predict_proba(cluster_test_df.drop(['label'], 1))
                        if np.size(scores, 1) >= 2:  # checks number of columns
                            scores = scores[:, 1]
                except (NotImplementedError, KeyError):
                    if hasattr(classifier[cluster], 'decision_function'):
                        scores = classifier[cluster].decision_function(cluster_test_df.drop(['label'], 1).values)
                    else:
                        scores = classifier[cluster].predict_proba(cluster_test_df.drop(['label'], 1).values)
                        try:
                            if np.size(scores, 1) >= 2:  # checks number of columns
                                scores = scores[:, 1]
                        except Exception as exception:
                            pass
                auc += get_auc(cluster_targets_df, scores)
            try:
                cluster_test_df['predicted'] = classifier[cluster].predict(cluster_test_df.drop(['label'], 1))
            except (NotImplementedError, KeyError):
                cluster_test_df['predicted'] = classifier[cluster].predict(cluster_test_df.drop(['label'], 1).values)

            results_df = results_df.append(cluster_test_df)

    if is_binary_classifier or max([len(set(t['label'])) for _, t in test_data.items()]) <= 2:
        auc = float(auc) / non_empty_clusters
    else:
        pass  # TODO: check if AUC is ok for multiclass, otherwise implement

    return results_df, auc


def predict(job: Job, data: DataFrame) -> Any:
    """Returns the predicted results whit classification

    :param job:
    :param data:
    :return:
    """
    data = data.drop(['trace_id'], 1)
    clusterer = Clustering.load_model(job)
    data = clusterer.cluster_data(data)

    classifier = joblib.load(job.predictive_model.model_path)

    non_empty_clusters = clusterer.n_clusters

    result = None

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = data[cluster]
        if cluster_test_df.empty:
            non_empty_clusters -= 1
        else:
            try:
                result = classifier[cluster].predict(cluster_test_df.drop(['label'], 1))
            except (NotImplementedError, KeyError):
                result = classifier[cluster].predict(cluster_test_df.drop(['label'], 1).values)

    return result


def predict_proba(job: Job, data: DataFrame) -> Any:
    """Returns the probability of predicted results whit classification

    :param job:
    :param data:
    :return:
    """
    data = data.drop(['trace_id'], 1)
    clusterer = Clustering.load_model(job)
    data = clusterer.cluster_data(data)

    classifier = joblib.load(job.predictive_model.model_path)

    non_empty_clusters = clusterer.n_clusters

    result = None

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = data[cluster]
        if cluster_test_df.empty:
            non_empty_clusters -= 1
        else:
            try:
                result = classifier[cluster].predict_proba(cluster_test_df.drop(['label'], 1))
            except (NotImplementedError, KeyError):
                result = classifier[cluster].predict_proba(cluster_test_df.drop(['label'], 1).values)

    return result


def _prepare_results(results_df: DataFrame, auc: int) -> dict:
    """Calculates and returns the results

    :param results_df:
    :param auc:
    :return:
    """
    actual = results_df['label'].values
    predicted = results_df['predicted'].values

    row = calculate_results_classification(actual, predicted)
    row['auc'] = auc
    return row


def _drop_columns(df: DataFrame) -> DataFrame:
    """Drops the column trace_id of given DataFrame

    :param df:
    :return:
    """
    df = df.drop('trace_id', 1)
    return df


def _choose_classifier(job: Job):
    """Chooses the classifier predictive method using the given job configuration

    :param job:
    :return:
    """
    method, config = get_method_config(job)
    config.pop('classification_method', None)
    logger.info("Using method {} with config {}".format(method, config))
    if method == ClassificationMethods.KNN.value:
        classifier = KNeighborsClassifier(**config)
    elif method == ClassificationMethods.RANDOM_FOREST.value:
        classifier = RandomForestClassifier(**config)
    elif method == ClassificationMethods.DECISION_TREE.value:
        classifier = DecisionTreeClassifier(**config)
    elif method == ClassificationMethods.XGBOOST.value:
        classifier = XGBClassifier(**config)
    elif method == ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value:
        classifier = MultinomialNB(**config)
    elif method == ClassificationMethods.ADAPTIVE_TREE.value:
        classifier = HAT(**config)
    elif method == ClassificationMethods.HOEFFDING_TREE.value:
        classifier = HoeffdingTree(**config)
    elif method == ClassificationMethods.SGDCLASSIFIER.value:
        classifier = SGDClassifier(**config)
    elif method == ClassificationMethods.PERCEPTRON.value:
        classifier = Perceptron(**config)
    elif method == ClassificationMethods.NN.value:
        config['encoding'] = job.encoding.value_encoding
        config['is_binary_classifier'] = _check_is_binary_classifier(job.labelling.type)
        classifier = NNClassifier(**config)
    else:
        raise ValueError("Unexpected classification method {}".format(method))
    return classifier


def _check_is_binary_classifier(label_type: str) -> bool:
    if label_type in [LabelTypes.REMAINING_TIME.value, LabelTypes.ATTRIBUTE_NUMBER.value, LabelTypes.DURATION.value]:
        return True
    if label_type in [LabelTypes.NEXT_ACTIVITY.value, LabelTypes.ATTRIBUTE_STRING.value]:
        return False
    raise ValueError("Label type {} not supported".format(label_type))
