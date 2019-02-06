from math import sqrt

import numpy as np
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score

from encoders.label_container import LabelContainer, REMAINING_TIME


def calculate_results_classification(actual: list, predicted: list) -> dict:
     return {**{'f1score': _get_f1(actual, predicted),
                'acc': accuracy_score(actual, predicted),
                'precision': _get_precision(actual, predicted),
                'recall': _get_recall(actual, predicted)},
             **get_confusion_matrix(actual, predicted)}


def get_confusion_matrix(actual, predicted) -> dict:
    tn, fp, fn, tp = '--', '--', '--', '--'
    actual_set = list(sorted(set(actual)))
    if len(actual_set) <= 2:
        if not isinstance(actual_set[0], bool) and not isinstance(actual_set[0], np.bool_):
            actual = [el == actual_set[0] for el in actual]
            predicted = [el == actual_set[0] for el in predicted]

        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()

    return {'true_positive': tp,
            'true_negative': tn,
            'false_negative': fn,
            'false_positive': fp}


def _get_f1(actual, predicted) -> float:
    try:
        f1score = f1_score(actual, predicted, average='macro')
    except ZeroDivisionError:
        f1score = 0
    return f1score


def _get_recall(actual, predicted) -> float:
    try:
        recall = recall_score(actual, predicted, average='macro')
    except ZeroDivisionError:
        recall = 0
    return recall


def _get_precision(actual, predicted) -> float:
    try:
        precision = precision_score(actual, predicted, average='macro')
    except ZeroDivisionError:
        precision = 0
    return precision


def calculate_auc(actual, scores, auc: int):
    if scores.shape[1] == 1:
        auc += 0
    else:
        try:
            auc += roc_auc_score(actual, scores[:, 1])
        except Exception:
            pass
    return auc


def calculate_results_regression(df: DataFrame, label: LabelContainer):
    if label.type == REMAINING_TIME:
        # TODO is the remaining time in seconds or hours?
        df['label'] = df['label'] / 3600
        df['prediction'] = df['prediction'] / 3600
    rmse = sqrt(mean_squared_error(df['label'], df['prediction']))
    mae = mean_absolute_error(df['label'], df['prediction'])
    rscore = r2_score(df['label'], df['prediction'])
    mape = _mean_absolute_percentage_error(df['label'], df['prediction'])

    row = {'rmse': rmse, 'mae': mae, 'rscore': rscore, 'mape': mape}
    return row


def _mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if 0 in y_true:
        return -1
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
