import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope

from core.constants import *


def _get_space(job: dict) -> dict:
    method_conf_name = "{}.{}".format(job['type'], job['method'])
    return HYPEROPT_SPACE_MAP[method_conf_name]()


def _classification_random_forest() -> dict:
    return {
        'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
        'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
        'max_features': hp.uniform('max_features', 0.0, 1.0)
    }


def _classification_knn() -> dict:
    return {
        'n_neighbors': hp.choice('n_neighbors', np.arange(1, 20, dtype=int)),
        'weights': hp.choice('weights', ['uniform', 'distance']),
    }


def _classification_decision_tree() -> dict:
    return {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
        'min_samples_split': hp.choice('min_samples_split', np.arange(2, 10, dtype=int)),
        'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 10, dtype=int)),
    }


def _classification_xgboost() -> dict:
    return {
        'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 30, 1)),
    }


def _classification_incremental_naive_bayes() -> dict:
    return {
        'alpha': hp.uniform('alpha', 0, 10),
        'fit_prior': True
    }


def _classification_incremental_adaptive_tree() -> dict:
    return {
        'grace_period': hp.uniform('grace_period', 1, 5),
        'split_criterion': hp.choice('split_criterion', ['gini', 'info_gain']),
        'split_confidence': hp.uniform('split_confidence', .0000005, .000001),
        'tie_threshold': hp.uniform('tie_threshold', .1, .6),
        # 'binary_split': hp.choice('binary_split', [ True, False ]),
        # 'stop_mem_management': hp.choice('stop_mem_management', [ True, False ]),
        'remove_poor_atts': hp.choice('remove_poor_atts', [True, False]),
        # 'no_preprune': hp.choice('no_preprune', [ True, False ]),
        'leaf_prediction': hp.choice('leaf_prediction', ['mc', 'nb', 'nba']),
        'nb_threshold': hp.uniform('nb_threshold', 0.2, 0.6)
    }


def _classification_incremental_hoeffding_tree() -> dict:
    return {
        'grace_period': hp.uniform('grace_period', 3, 8),
        'split_criterion': hp.choice('split_criterion', ['gini', 'info_gain']),
        'split_confidence': hp.uniform('split_confidence', .0000005, .0000009),
        'tie_threshold': hp.uniform('tie_threshold', .4, .8),
        # 'binary_split': hp.choice('binary_split', [ True, False ]),
        # 'stop_mem_management': hp.choice('stop_mem_management', [ True, False ]),
        'remove_poor_atts': hp.choice('remove_poor_atts', [True, False]),
        # 'no_preprune': hp.choice('no_preprune', [ True, False ]),
        'leaf_prediction': hp.choice('leaf_prediction', ['mc', 'nb', 'nba']),
        'nb_threshold': hp.uniform('nb_threshold', 0.1, 0.5)
    }


def _classification_incremental_sgd_classifier() -> dict:
    return {
        'loss': hp.choice('loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss',
                                   'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
        'penalty': hp.choice('penalty', [None, 'l1', 'l2', 'elasticnet']),
        'alpha': hp.uniform('alpha', 0.0001, 0.5),
        'l1_ratio': hp.uniform('l1_ratio', 0.15, 1.0),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'tol': hp.uniform('tol', 1e-3, 0.5),
        'epsilon': hp.uniform('epsilon', 1e-3, 0.5),
        'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
        'eta0': scope.int(hp.quniform('eta0', 4, 30, 1)),
        'power_t': hp.uniform('power_t', 0.5, 0.1),
        'early_stopping': hp.choice('early_stopping', [True, False]),
        'n_iter_no_change': scope.int(hp.quniform('n_iter_no_change', 5, 30, 5)),
        'validation_fraction': 0.1,
        'average': hp.choice('average', [True, False])
    }


def _classification_incremental_perceptron() -> dict:
    return {
        'penalty': hp.choice('penalty', [None, 'l1', 'l2', 'elasticnet']),
        'alpha': hp.uniform('alpha', 0.0001, 0.5),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'tol': hp.uniform('tol', 1e-3, 0.5),
        'shuffle': hp.choice('shuffle', [True, False]),
        'eta0': scope.int(hp.quniform('eta0', 4, 30, 1)),
        'early_stopping': hp.choice('early_stopping', [True, False]),
        'validation_fraction': 0.1,
        'n_iter_no_change': scope.int(hp.quniform('n_iter_no_change', 5, 30, 5))
    }


def _regression_random_forest() -> dict:
    return {
        'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
        'max_features': hp.uniform('max_features', 0.0, 1.0),
        'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
    }


def _regression_lasso() -> dict:
    return {
        'alpha': hp.uniform('alpha', 0.01, 2.0),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'normalize': hp.choice('normalize', [True, False])
    }


def _regression_linear() -> dict:
    return {
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'normalize': hp.choice('normalize', [True, False])
    }


def _regression_xgboost() -> dict:
    return {
        'max_depth': scope.int(hp.quniform('max_depth', 3, 100, 1)),
        'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
    }

def _clusterisation_dbscan() -> dict:
    return {
        'eps': hp.uniform('eps', 0.0, 1.0),
        'min_samples': hp.choice('min_samples', np.arange(2, 7, dtype=int)),
        'metric': hp.choice('metric', ['euclidean', 'manhattan', 'precomputed'])
    }

def _clusterisation_agglomerative() -> dict:
    return {
        'n_clusters': hp.choice('n_clusters', np.arange(3, 5, dtype=int)),
        'affinity': hp.choice('affinity', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']),
        'linkage': hp.choice('linkage' ['ward', 'complete', 'average', 'single'])
    }

HYPEROPT_SPACE_MAP = {
    CLASSIFICATION_RANDOM_FOREST: _classification_random_forest,
    CLASSIFICATION_KNN: _classification_knn,
    CLASSIFICATION_XGBOOST: _classification_xgboost,
    CLASSIFICATION_DECISION_TREE: _classification_decision_tree,
    CLASSIFICATION_MULTINOMIAL_NAIVE_BAYES: _classification_incremental_naive_bayes,
    CLASSIFICATION_ADAPTIVE_TREE: _classification_incremental_adaptive_tree,
    CLASSIFICATION_HOEFFDING_TREE: _classification_incremental_hoeffding_tree,
    CLASSIFICATION_SGDC: _classification_incremental_sgd_classifier,
    CLASSIFICATION_PERCEPTRON: _classification_incremental_perceptron,
    REGRESSION_RANDOM_FOREST: _regression_random_forest,
    REGRESSION_XGBOOST: _regression_xgboost,
    REGRESSION_LASSO: _regression_lasso,
    REGRESSION_LINEAR: _regression_linear,
    CLUSTERISATION_DBSCAN: _clusterisation_dbscan,
    CLUSTERISATION_AGGLOMERATIVE: _clusterisation_agglomerative
}
