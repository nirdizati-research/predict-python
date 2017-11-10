"""Common methods for all training methods"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from core.constants import SIMPLE_INDEX, BOOLEAN, FREQUENCY, KNN, RANDOM_FOREST, DECISION_TREE, NEXT_ACTIVITY
from encoders.boolean_frequency import boolean, frequency
from encoders.simple_index import simple_index
from logs.file_service import get_logs


def encode(job):
    """Get encoded data frame"""
    # print job.encoding
    log = get_logs('log_cache/general_example.xes')[0]
    if job.encoding == BOOLEAN:
        return boolean(log)
    elif job.encoding == FREQUENCY:
        return frequency(log)
    elif job.encoding == SIMPLE_INDEX:
        return simple_index(log, prefix_length=1, next_activity=(job.type == NEXT_ACTIVITY))


def calculate_results(prediction, actual):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for i in range(0, len(actual)):
        if actual[i]:
            if actual[i] == prediction[i]:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if actual[i] == prediction[i]:
                true_negative += 1
            else:
                false_negative += 1

    # print 'TP: ' + str(true_positive) + ' FP: ' + str(false_positive) + ' FN: ' + str(false_negative)
    try:
        precision = float(true_positive) / (true_positive + false_positive)

        recall = float(true_positive) / (true_positive + false_negative)
        f1score = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1score = 0

    acc = float(true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
    return f1score, acc


def choose_classifier(job):
    clf = None
    if job.classification == KNN:
        clf = KNeighborsClassifier()
    elif job.classification == RANDOM_FOREST:
        clf = RandomForestClassifier()
    elif job.classification == DECISION_TREE:
        clf = DecisionTreeClassifier()
    return clf


def fast_slow_encode(df, label, threshold):
    if threshold == "default":
        threshold_ = df[label].mean()
    else:
        threshold_ = float(threshold)
    df['actual'] = df[label] < threshold_
    return df
