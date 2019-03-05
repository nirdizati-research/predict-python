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
from src.jobs.models import JobTypes, Job
from src.labelling.models import LabelTypes
from src.predictive_model.classification.custom_classification_models import NNClassifier
from src.predictive_model.classification.models import ClassificationMethods, Classification
from src.utils.result_metrics import calculate_results_classification, get_auc

pd.options.mode.chained_assignment = None


def classification(training_df: DataFrame, test_df: DataFrame, job: Job) -> (dict, dict):
    """main classification entry point

    train and tests the classifier using the provided data

    :param training_df: training DataFrame
    :param test_df: testing DataFrame
    :param job: job configuration
    :return: predictive_model scores and split

    """
    train_data, test_data = _drop_columns(training_df, test_df)

    model_split = _train(job, train_data, _choose_classifier(job))
    results_df, auc = _test(model_split, test_data, evaluation=True,
                            is_binary_classifier=_check_is_binary_classifier(job.labelling.type))

    results = _prepare_results(results_df, auc)

    # TODO save predictive_model more wisely
    model_split['type'] = job['clustering']

    return results, model_split


def classification_single_log(input_df: DataFrame, model: dict) -> dict:
    """single log classification

    classifies a single log using the provided TODO: complete

    :param input_df: input DataFrame
    :param model: TODO: complete
    :return: predictive_model scores

    """
    results = dict()
    split = model['split']
    results['label'] = input_df['label']

    # TODO load predictive_model more wisely
    model_split = dict()
    model_split['clusterer'] = joblib.load(split['clusterer_path'])
    model_split['classifier'] = joblib.load(split['model_path'])
    result, _ = _test(model_split, input_df, evaluation=False,
                      is_binary_classifier=_check_is_binary_classifier(model['label'].type))
    results['prediction'] = result['predicted']
    return results


def update_and_test(training_df: DataFrame, test_df: DataFrame, job: Job):
    train_data, test_data = _drop_columns(training_df, test_df)

    #TODO load previously used data structure to check input conformance
    # Encoding.objects.filter()

    model_split = _update(job, train_data, _choose_classifier(job))

    results_df, auc = _test(model_split, test_data, evaluation=True,
                            is_binary_classifier=_check_is_binary_classifier(job.labelling.type))

    results = _prepare_results(results_df, auc)

    # TODO mmh not sure
    model_split['type'] = job['clustering']

    job['type'] = JobTypes.UPDATE.value

    return results, model_split


def _train(job: Job, train_data: DataFrame, classifier: ClassifierMixin) -> dict:
    clusterer = Clustering(job)
    models = dict()

    clusterer.fit(train_data.drop('label', 1))

    train_data = clusterer.cluster_data(train_data)

    for cluster in range(clusterer.n_clusters):

        cluster_train_df = train_data[cluster]
        if not cluster_train_df.empty:
            cluster_targets_df = DataFrame(cluster_train_df['label'])
            try:
                classifier.fit(cluster_train_df.drop('label', 1), cluster_targets_df.values.ravel())
            except NotImplementedError:
                classifier.partial_fit(cluster_train_df.drop('label', 1), cluster_targets_df.values.ravel())
            except Exception as exception:
                raise exception

            models[cluster] = classifier
            try:
                classifier = clone(classifier)
            except TypeError:
                classifier = clone(classifier, safe=False)
                classifier.reset()

    return {'clusterer': clusterer, 'classifier': models}


def _update(job: Job, data: DataFrame, models) -> dict:
    clusterer = Clustering.load_model(job)

    update_data = clusterer.cluster_data(data)

    for cluster in range(clusterer.n_clusters):
        x = update_data[cluster]
        if not x.empty:
            y = x['label']

            models[cluster].partial_fit(x.drop('label', 1), y.values.ravel())

    return {'clusterer': clusterer, 'classifier': models}


def _test(model_split: dict, test_data: DataFrame, evaluation: bool, is_binary_classifier: bool) -> (DataFrame, float):
    clusterer = model_split['clusterer']
    classifier = model_split['classifier']

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
                if hasattr(classifier[cluster], 'decision_function'):
                    scores = classifier[cluster].decision_function(cluster_test_df.drop(['label'], 1))
                else:
                    scores = classifier[cluster].predict_proba(cluster_test_df.drop(['label'], 1))
                    if np.size(scores, 1) >= 2:  # checks number of columns
                        scores = scores[:, 1]
                auc += get_auc(cluster_targets_df, scores)
            cluster_test_df['predicted'] = classifier[cluster].predict(cluster_test_df.drop(['label'], 1))

            results_df = results_df.append(cluster_test_df)

    if is_binary_classifier or max([len(set(t['label'])) for _, t in test_data.items()]) <= 2:
        auc = float(auc) / non_empty_clusters
    else:
        pass  # TODO: check if AUC is ok for multiclass, otherwise implement

    return results_df, auc


def _prepare_results(results_df: DataFrame, auc: int) -> dict:
    actual = results_df['label'].values
    predicted = results_df['predicted'].values

    row = calculate_results_classification(actual, predicted)
    row['auc'] = auc
    return row


def _drop_columns(train_df: DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame):
    train_df = train_df.drop('trace_id', 1)
    test_df = test_df.drop('trace_id', 1)
    return train_df, test_df


def _choose_classifier(job: Job):
    if job.type == JobTypes.UPDATE.value:
        classifier = _load_model(job.incremental_train)
        # TODO: check if this instruction still makes sense
        assert classifier[0].__class__.__name__ == job.method # are we updating a predictive_model with its own methods?
    else:
        method, config = get_method_config(job)
        print("Using method {} with config {}".format(method, config))
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


def _load_model(base_model: Classification):
    # classifier = PredModels.objects.filter(id=incremental_base_model)
    # assert len(classifier) == 1  # asserting that the used id is unique
    # classifier_details = classifier[0]
    # classifier = ModelSplit.objects.filter(id=classifier_details.split_id)
    # assert len(classifier) == 1
    # classifier = classifier[0]
    # classifier = joblib.load(classifier.model_path)
    # return classifier
    #     TODO: fix this
    return None


def _check_is_binary_classifier(label_type: str) -> bool:
    if label_type in [LabelTypes.REMAINING_TIME.value, LabelTypes.ATTRIBUTE_NUMBER.value, LabelTypes.DURATION.value]:
        return True
    if label_type in [LabelTypes.NEXT_ACTIVITY.value, LabelTypes.ATTRIBUTE_STRING.value]:
        return False
    raise ValueError("Label type not supported", label_type)
