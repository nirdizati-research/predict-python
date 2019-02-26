import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from matplotlib import pyplot

#calculating cross entropy
def cross_entropy(predicted_probabilities,actual_classes):
    result = (np.sum(-actual_classes * np.log2(predicted_probabilities[:, 1]) - (1 - actual_classes) * np.log2(1 - predicted_probabilities[:, 1])))/len(actual_classes)
    return result

#applying logistic calibration on model
def calibrate_classifier(training_df, test_df, model):
    train_data, test_data, original_test_data = prep_data(training_df, test_df)

    calibrated_ml = CalibratedClassifierCV(model, cv=2, method='sigmoid')
    calibrated_ml.fit(train_data.drop('label', 1), train_data['label'])
    calibrated_ml_prob = calibrated_ml.predict_proba(test_data.drop('label', 1))

    print(cross_entropy(calibrated_ml_prob, train_data['label']))
    return calibrated_ml

##make plots???
def make_probabilities(calibrated_probability, test_data):
    fop, mpv = calibration_curve(test_data['label'], calibrated_probability, n_bins=10, normalize=True)
    # plot perfectly calibrated
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot calibrated reliability
    pyplot.plot(mpv, fop, marker='.')
    pyplot.show()

def prep_data(training_df, test_df):
    train_data = training_df
    test_data = test_df

    original_test_data = test_data

    test_data = test_data.drop(['trace_id', 'label'], 1)
    train_data = train_data.drop('trace_id', 1)
    return train_data, test_data, original_test_data
