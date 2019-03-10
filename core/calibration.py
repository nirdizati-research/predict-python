import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from matplotlib import pyplot

def cross_entropy(predicted_probabilities,actual_classes):
    """
    Calculate cross entropy
    """

    result = (np.sum(-actual_classes * np.log2(predicted_probabilities[:, 1]) - (1 - actual_classes) * np.log2(1 - predicted_probabilities[:, 1])))/len(actual_classes)
    return result

def calibrate_classifier(training_df, test_df, model):
    """
    Applying logistic calibration on model

    :param training_df: training dataset
    :param test_df: test dataset
    :param model: model to be calibrated
    :return: calibrated model
    """

    train_data, test_data, original_test_data = prep_data(training_df, test_df)

    calibrated_ml = CalibratedClassifierCV(model, cv=2, method='sigmoid')
    calibrated_ml.fit(train_data.drop('label', 1), train_data['label'])
    calibrated_ml_prob = calibrated_ml.predict_proba(test_data.drop('label', 1))

    print(cross_entropy(calibrated_ml_prob, train_data['label']))
    return calibrated_ml

def make_probabilities(calibrated_probability, test_data):
    """
    Create plots of calibrated probabilities
    """

    fop, mpv = calibration_curve(test_data['label'], calibrated_probability, n_bins=10, normalize=True)
    # plot perfectly calibrated
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot calibrated reliability
    pyplot.plot(mpv, fop, marker='.')
    pyplot.show()

def prep_data(training_df, test_df):
    """
    Split features from labels in the dataset

    :param training_df: training dataset
    :param test_df: test dataset
    :return: training dataset, test dataset and original dataset
    """

    train_data = training_df
    test_data = test_df

    original_test_data = test_data

    test_data = test_data.drop(['trace_id', 'label'], 1)
    train_data = train_data.drop('trace_id', 1)
    return train_data, test_data, original_test_data
