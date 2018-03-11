"""Common methods for all training methods"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from core.constants import KNN, RANDOM_FOREST, DECISION_TREE


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
    row = {'f1score': f1score, 'acc': acc, 'true_positive': true_positive, 'true_negative': true_negative,
           'false_negative': false_negative, 'false_positive': false_positive, 'precision': precision, 'recall': recall}
    return row


def choose_classifier(class_type: str):
    if class_type == KNN:
        clf = KNeighborsClassifier()
    elif class_type == RANDOM_FOREST:
        clf = RandomForestClassifier()
    elif class_type == DECISION_TREE:
        clf = DecisionTreeClassifier()
    else:
        raise ValueError("Unexpected classification method {}".format(class_type))
    return clf


# TODO deprecate
def fast_slow_encode(df, label, threshold):
    if threshold == "default":
        threshold_ = df[label].mean()
    else:
        threshold_ = float(threshold)
    df['actual'] = df[label] < threshold_
    return df


def fast_slow_encode2(training_df, test_df, label: str, threshold: float):
    if threshold == "default":
        complete_df = training_df.append(test_df)
        threshold_ = complete_df[label].mean()
    else:
        threshold_ = float(threshold)
    training_df['actual'] = training_df[label] < threshold_
    test_df['actual'] = test_df[label] < threshold_
    return training_df, test_df
