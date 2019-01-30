from math import sqrt

from pandas import DataFrame
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
import numpy as np
from encoders.label_container import LabelContainer, REMAINING_TIME


def calculate_results_binary_classification(actual: list, predicted: list):
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


def calculate_results_multiclass_classification(actual: list, predicted: list):
    # average is needed as these are multi-label lists
    # print(classification_report(actual, predicted))
    acc = accuracy_score(actual, predicted)
    f1score = f1_score(actual, predicted, average='macro')
    precision = precision_score(actual, predicted, average='macro')
    recall = recall_score(actual, predicted, average='macro')

    if len(set(actual + predicted)) == 2:
        row = calculate_results_binary_classification([el == 'true' for el in actual], [el == 'true' for el in predicted])
    else:
        row = {'f1score': f1score, 'acc': acc, 'precision': precision, 'recall': recall}
    return row


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
        # TODO are remaining time in seconds or hours?
        df['label'] = df['label'] / 3600
        df['prediction'] = df['prediction'] / 3600
    rmse = sqrt(mean_squared_error(df['label'], df['prediction']))
    mae = mean_absolute_error(df['label'], df['prediction'])
    rscore = r2_score(df['label'], df['prediction'])
    mape = mean_absolute_percentage_error(df['label'], df['prediction'])

    row = {'rmse': rmse, 'mae': mae, 'rscore': rscore, 'mape': mape}
    return row


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
