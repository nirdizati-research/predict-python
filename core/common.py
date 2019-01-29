"""Common methods for all training methods"""

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from core.constants import KNN, RANDOM_FOREST, DECISION_TREE, XGBOOST


def calculate_results(actual, predicted):
    conf_matrix = confusion_matrix(actual, predicted, labels=[False, True])

    try:
        precision = precision_score(actual, predicted)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = recall_score(actual, predicted)
    except ZeroDivisionError:
        recall = 0

    try:
        f1score = f1_score(actual, predicted)
    except ZeroDivisionError:
        f1score = 0

    acc = accuracy_score(actual, predicted)
    row = {'f1score': f1score, 'acc': acc, 'true_positive': conf_matrix[1][1], 'true_negative': conf_matrix[0][0],
           'false_negative': conf_matrix[0][1], 'false_positive': conf_matrix[1][0], 'precision': precision,
           'recall': recall}
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
    else:
        raise ValueError("Unexpected classification method {}".format(method))
    return clf


def get_method_config(job: dict):
    method = job['method']
    method_conf_name = "{}.{}".format(job['type'], method)
    config = job[method_conf_name]
    return method, config
