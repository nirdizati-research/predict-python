from math import sqrt

import numpy as np
from distance import nlevenshtein
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score

from src.labelling.models import LabelTypes, Labelling


def calculate_results_classification(actual: list, predicted: list) -> dict:
    """Returns the results of classification predictive model

    :param actual:
    :param predicted:
    :return:
    """
    return {**{'f1score': _get_f1(actual, predicted),
               'acc': accuracy_score(actual, predicted),
               'precision': _get_precision(actual, predicted),
               'recall': _get_recall(actual, predicted)},
            **get_confusion_matrix(actual, predicted)}


def calculate_results_time_series_prediction(actual: ndarray, predicted: ndarray) -> dict:
    """Returns the results of classification predictive model

    :param actual:
    :param predicted:
    :return:
    """
    return {}


def get_confusion_matrix(actual, predicted) -> dict:
    """Returns the confusion matrix of model

    :param actual:
    :param predicted:
    :return:
    """
    true_negatives, false_positives, false_negatives, true_positives = '--', '--', '--', '--'
    actual_set = list(sorted(set(actual)))
    if len(actual_set) <= 2:
        if not isinstance(actual_set[0], bool) and not isinstance(actual_set[0], np.bool_):
            actual = [el == actual_set[0] for el in actual]
            predicted = [el == actual_set[0] for el in predicted]

        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(actual, predicted,
                                                                                            labels=[False,
                                                                                                    True]).ravel()

    return {'true_positive': true_positives,
            'true_negative': true_negatives,
            'false_negative': false_negatives,
            'false_positive': false_positives}


def _get_f1(actual, predicted) -> float:
    """Returns the f1_score of model

    :param actual:
    :param predicted:
    :return:
    """
    try:
        f1score = f1_score(actual, predicted, average='macro')
    except ZeroDivisionError:
        f1score = 0
    return f1score


def _get_recall(actual, predicted) -> float:
    """Returns the recall of the model

    :param actual:
    :param predicted:
    :return:
    """
    try:
        recall = recall_score(actual, predicted, average='macro')
    except ZeroDivisionError:
        recall = 0
    return recall


def _get_precision(actual, predicted) -> float:
    """Returns the precision of model

    :param actual:
    :param predicted:
    :return:
    """
    try:
        precision = precision_score(actual, predicted, average='macro')
    except ZeroDivisionError:
        precision = 0
    return precision


def calculate_nlevenshtein(actual: ndarray, predicted: ndarray) -> float:
    """Returns the nlevenshtein of model

    :param actual:
    :param predicted:
    :return:
    """
    distances = []

    for row in range(actual.shape[0]):
        distances.append(nlevenshtein(np.array2string(actual[row]), np.array2string(predicted[row])))

    return float(np.mean(distances))


def get_auc(actual, scores) -> float:
    """Returns the roc_auc_score of model

    :param actual:
    :param scores:
    :return:
    """
    try:
        auc = roc_auc_score(actual, scores)
    except ValueError:
        auc = 0
    return auc


def calculate_auc(actual, scores, auc: int) -> float:
    """Calculate and returns the roc_auc_score of model

    :param actual:
    :param scores:
    :param auc:
    :return:
    """
    if scores.shape[1] == 1:
        auc += 0
    else:
        try:
            auc += roc_auc_score(actual, scores[:, 1])
        except Exception:
            pass
    return auc


def _prepare_results(input_df: DataFrame, label: Labelling) -> dict:
    """Prepares the result

    :param input_df:
    :param label:
    :return:
    """
    if label.type == LabelTypes.REMAINING_TIME.value:
        # TODO is the remaining time in seconds or hours?
        input_df['label'] = input_df['label'] / 3600
        input_df['predicted'] = input_df['predicted'] / 3600
    rmse = sqrt(mean_squared_error(input_df['label'], input_df['predicted']))
    mae = mean_absolute_error(input_df['label'], input_df['predicted'])
    rscore = r2_score(input_df['label'], input_df['predicted'])
    mape = _mean_absolute_percentage_error(input_df['label'], input_df['predicted'])

    row = {'rmse': rmse, 'mae': mae, 'rscore': rscore, 'mape': mape}
    return row


def _mean_absolute_percentage_error(y_true, y_pred):
    """Calculates and returns the mean absolute percentage error

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if 0 in y_true:
        return -1
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
