"""
hyperopt search spaces for each prediction method
"""

import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope

from src.jobs.models import Job
from src.predictive_model.classification.models import CLASSIFICATION_RANDOM_FOREST, CLASSIFICATION_KNN, \
    CLASSIFICATION_XGBOOST, CLASSIFICATION_MULTINOMIAL_NAIVE_BAYES, CLASSIFICATION_DECISION_TREE, \
    CLASSIFICATION_ADAPTIVE_TREE, \
    CLASSIFICATION_HOEFFDING_TREE, CLASSIFICATION_SGDC, CLASSIFICATION_PERCEPTRON, CLASSIFICATION_NN
from src.predictive_model.regression.models import REGRESSION_RANDOM_FOREST, REGRESSION_XGBOOST, REGRESSION_LASSO, \
    REGRESSION_LINEAR, REGRESSION_NN
from src.predictive_model.time_series_prediction.models import TIME_SERIES_PREDICTION_RNN


def _get_space(job: Job) -> dict:
    method_conf_name = "{}.{}".format(job.predictive_model.predictive_model, job.predictive_model.prediction_method)
    return HYPEROPT_SPACE_MAP[method_conf_name]()


def _classification_random_forest() -> dict:
    return {
        'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
        'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', 'auto', None])
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
        'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),
        'alpha': hp.uniform('alpha', 0.0001, 0.5),
        'l1_ratio': hp.uniform('l1_ratio', 0.15, 1.0),
        'fit_intercept': hp.choice('fit_intercept', [True, False]),
        'tol': hp.uniform('tol', 1e-3, 0.5),
        'epsilon': hp.uniform('epsilon', 1e-3, 0.5),
        'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
        'eta0': scope.int(hp.quniform('eta0', 4, 30, 1)),
        'power_t': hp.uniform('power_t', 0.3, 0.7),
        # 'early_stopping': hp.choice('early_stopping', [True, False]), #needs to be false with partial_fit
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
        # 'early_stopping': hp.choice('early_stopping', [True, False]), #needs to be false with partial_fit
        'validation_fraction': 0.1,
        'n_iter_no_change': scope.int(hp.quniform('n_iter_no_change', 5, 30, 5))
    }


def _classification_nn() -> dict:
    return {
        'hidden_layers': hp.quniform('hidden_layers', 1, 10),
        'hidden_units': hp.quniform('hidden_layers', 1, 100),
        'activation_function': hp.choice('activation_function', ['sigmoid', 'tanh', 'relu']),
        'epochs': hp.quniform('epochs', 1, 50),
        'dropout_rate': hp.uniform('dropout_rate', 0, 1)
    }


def _regression_random_forest() -> dict:
    return {
        'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', 'auto', None]),
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


def _regression_nn() -> dict:
    return {
        'hidden_layers': hp.quniform('hidden_layers', 1, 10),
        'hidden_units': hp.quniform('hidden_layers', 1, 100),
        'activation_function': hp.choice('activation_function', ['sigmoid', 'tanh', 'relu']),
        'epochs': hp.quniform('epochs', 1, 50),
        'dropout_rate': hp.uniform('dropout_rate', 0, 1)
    }


def _time_series_prediction_rnn() -> dict:
    raise NotImplementedError


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
    CLASSIFICATION_NN: _classification_nn,

    REGRESSION_RANDOM_FOREST: _regression_random_forest,
    REGRESSION_XGBOOST: _regression_xgboost,
    REGRESSION_LASSO: _regression_lasso,
    REGRESSION_LINEAR: _regression_linear,
    REGRESSION_NN: _regression_nn,

    TIME_SERIES_PREDICTION_RNN: _time_series_prediction_rnn
}
