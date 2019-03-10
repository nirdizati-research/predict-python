import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, confusion_matrix

from IPython.display import display

def tune_model(model, param_grid, X_train, y_train, cv=5, max_rows=10):
    """
    Tune model

    :param model: model to be tuned
    :param param_grid: dictionary with parameters names
    :param X_train: train dataset features
    :param y_train: train dataset labels
    :param cv: cross-validation generator
    :param max_rows: number of rows
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv, return_train_score=True, scoring=['accuracy','f1','roc_auc'],refit='accuracy')
    grid_search.fit(X_train, y_train)
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_column_filter(cv_results)
    cv_results = cv_results.sort_values('mean_test_accuracy',ascending=False)
    pd.set_option('max_rows',max_rows)
    display(cv_results)
    pd.reset_option('max_rows')
    return(grid_search)

def roc_plot(models, thresholded_models, X, y):
    """ Create ROC plot"""

    fig = plt.figure(figsize=(8,8))
    for model in models:
        probs = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, probs)
        fpr,tpr,thr = roc_curve(y,probs)
        plt.plot(fpr, tpr, label='{} (AUC={:.2f})'.format(str(model.estimator).split('(')[0], auc))
    for t_model in thresholded_models:
        y_pred = t_model.predict(X)
        tpr = np.sum(np.logical_and(y_pred, y == 1)) / np.sum(y == 1)
        fpr = np.sum(np.logical_and(y_pred, y == 0)) / np.sum(y == 0)
        plt.plot(fpr, tpr, 'x', label='{}'.format(str(t_model)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()

def cv_column_filter(df):
    return(df.loc[:,[c for c in df.columns
                     if (c[:11]=='mean_train_') or
                     (c[:10]=='mean_test_') or
                     (c[:6]=='param_')]])

def main(cv_results):
    cv_results = cv_column_filter(cv_results)
    cv_results = cv_results.sort_values('mean_test_score',ascending=False)
    pd.set_option('max_rows', 20)
    return cv_results
