import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope

from core.constants import *


def get_space(job: dict):
    method_conf_name = "{}.{}".format(job['type'], job['method'])
    return HYPEROPT_SPACE_MAP[method_conf_name]()


def _classification_random_forest():
    return {'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'max_features': hp.uniform('max_features', 0.0, 1.0)
            }


# test case dynamic max feature
def _classification_knn():
    return {
        'n_neighbors': hp.choice('n_neighbors', np.arange(1, 20, dtype=int)),
        'weights': hp.choice('weights', ['uniform', 'distance']),
    }


def _classification_decision_tree():
    return {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
        'min_samples_split': hp.choice('min_samples_split', np.arange(2, 10, dtype=int)),
        'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 10, dtype=int)),
    }


def _classification_xgboost():
    return {
        'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 30, 1)),
    }


def _regression_random_forest():
    return {
        'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
        'max_features': hp.uniform('max_features', 0.0, 1.0),
        'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
    }


def _regression_lasso():
    return {
        'alpha': hp.uniform('alpha', 0.01, 2.0),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'normalize': hp.choice('normalize', [True, False])
    }


def _regression_linear():
    return {
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'normalize': hp.choice('normalize', [True, False])
    }


def _regression_xgboost():
    return {
        'max_depth': scope.int(hp.quniform('max_depth', 0, 100, 1)),
        'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
    }


HYPEROPT_SPACE_MAP = {CLASSIFICATION_RANDOM_FOREST: _classification_random_forest,
                      CLASSIFICATION_KNN: _classification_knn, CLASSIFICATION_XGBOOST: _classification_xgboost,
                      CLASSIFICATION_DECISION_TREE: _classification_decision_tree,
                      REGRESSION_RANDOM_FOREST: _regression_random_forest, REGRESSION_XGBOOST: _regression_xgboost,
                      REGRESSION_LASSO: _regression_lasso, REGRESSION_LINEAR: _regression_linear}
