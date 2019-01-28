"""Common methods for all training methods"""

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from core.constants import KNN, RANDOM_FOREST, DECISION_TREE, XGBOOST, INCREMENTAL_NAIVE_BAYES, \
    INCREMENTAL_ADAPTIVE_TREE, INCREMENTAL_HOEFFDING_TREE
from skmultiflow.trees import HAT, HoeffdingTree


def calculate_results(prediction, actual):

    conf_matrix = confusion_matrix(actual, prediction)

    try:
        precision = precision_score(actual, prediction)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = recall_score(actual, prediction)
    except ZeroDivisionError:
        recall = 0

    try:
        f1score = f1_score(actual, prediction)
    except ZeroDivisionError:
        f1score = 0

    acc = accuracy_score(actual, prediction)
    row = {'f1score': f1score, 'acc': acc, 'true_positive': conf_matrix[1][1], 'true_negative': conf_matrix[0][0],
           'false_negative': conf_matrix[0][1], 'false_positive': conf_matrix[1][0], 'precision': precision, 'recall': recall}
    return row


def choose_classifier(job: dict):
    method, config = get_method_config(job)
    print("Using method {} with config {}".format(method, config))
    if method == KNN:
        clf = KNeighborsClassifier(**config)
    elif method == RANDOM_FOREST:
        clf = RandomForestClassifier(**config)
    elif method == DECISION_TREE:
        clf = DecisionTreeClassifier(**config)
    elif method == XGBOOST:
        clf = xgb.XGBClassifier(**config)
    elif method == INCREMENTAL_NAIVE_BAYES: #TODO check which is better as model (MultinomialNB, BernoulliNB, GaussianNB)
        clf = MultinomialNB(**config)
    elif method == INCREMENTAL_ADAPTIVE_TREE:
        clf = HAT(**config)
    elif method == INCREMENTAL_HOEFFDING_TREE:
        clf = HoeffdingTree(**config)
    else:
        raise ValueError("Unexpected classification method {}".format(method))
    return clf


def get_method_config(job: dict):
    method = job['method']
    method_conf_name = "{}.{}".format(job['type'], method)
    config = job[method_conf_name]
    return method, config


def add_actual(training_df, test_df):
    training_df['actual'] = training_df['label']
    test_df['actual'] = test_df['label']
    return training_df, test_df
