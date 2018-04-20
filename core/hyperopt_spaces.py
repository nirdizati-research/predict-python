import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope

from core.constants import *


def get_space(job: dict):
    method_conf_name = "{}.{}".format(job['type'], job['method'])
    return HYPEROPT_SPACE_MAP[method_conf_name]()


def _classification_random_forest():
    return {'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'criterion': hp.choice('criterion', ['gini', 'entropy']),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'min_samples_split': hp.choice('min_samples_split', np.arange(2, 10, dtype=int)),
            'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 10, dtype=int)),
            }


def _classification_knn():
    return {
        'n_neighbors': hp.choice('n_neighbors', np.arange(1, 20, dtype=int)),
        'weights': hp.choice('weights', ['uniform', 'distance']),
    }


HYPEROPT_SPACE_MAP = {CLASSIFICATION_RANDOM_FOREST: _classification_random_forest,
                      CLASSIFICATION_KNN: _classification_knn,
                      # CLASSIFICATION_DECISION_TREE: _classification_decision_tree,
                      # REGRESSION_RANDOM_FOREST: _regression_random_forest,
                      # REGRESSION_LASSO: _regression_lasso, REGRESSION_LINEAR: _regression_linear,
                      # NEXT_ACTIVITY_RANDOM_FOREST: _classification_random_forest,
                      # NEXT_ACTIVITY_KNN: _classification_knn,
                      # NEXT_ACTIVITY_DECISION_TREE: _classification_decision_tree}
                      }