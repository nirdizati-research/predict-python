"""
default configurations for the prediction methods
"""
from src.core.constants import CLASSIFICATION_RANDOM_FOREST, CLASSIFICATION_KNN, CLASSIFICATION_DECISION_TREE, \
    CLASSIFICATION_XGBOOST, CLASSIFICATION_MULTINOMIAL_NAIVE_BAYES, CLASSIFICATION_NN, CLASSIFICATION_ADAPTIVE_TREE, \
    CLASSIFICATION_HOEFFDING_TREE, CLASSIFICATION_SGDC, CLASSIFICATION_PERCEPTRON, REGRESSION_RANDOM_FOREST, \
    REGRESSION_XGBOOST, REGRESSION_LASSO, REGRESSION_LINEAR, REGRESSION_NN, TIME_SERIES_PREDICTION_RNN, \
    UPDATE_INCREMENTAL_NAIVE_BAYES, UPDATE_INCREMENTAL_ADAPTIVE_TREE, UPDATE_INCREMENTAL_HOEFFDING_TREE


def clustering_kmeans():
    return {
        'n_clusters': 8,
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300,
        'tol': 1e-4,
        'precompute_distances': 'auto',
        'random_state': None,
        'copy_x': None,
        'algorithm': None
    }


def classification_random_forest():
    return {
        'n_estimators': 10,
        'max_depth': None,
        'max_features': 'auto',
        'n_jobs': -1,
        'random_state': 21
    }


def classification_knn():
    return {
        'n_neighbors': 5,
        'n_jobs': -1,
        'weights': 'uniform'
    }


def classification_decision_tree():
    return {
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 21
    }


def classification_incremental_naive_bayes():
    return {
        'alpha': 1.0,
        'fit_prior': True,
        'class_prior': None
    }


def classification_incremental_adaptive_tree():
    return {
        'max_byte_size': 33554432,
        'memory_estimate_period': 1000000,
        'grace_period': 200,
        'split_criterion': 'info_gain',
        'split_confidence': .0000001,
        'tie_threshold': .05,
        'binary_split': False,
        'stop_mem_management': False,
        'remove_poor_atts': False,
        'no_preprune': False,
        'leaf_prediction': 'nba',
        'nb_threshold': 0,
        'nominal_attributes': []  # <-- if this is empty assume all attributes are numerical
    }


def classification_incremental_hoeffding_tree():
    return {
        'max_byte_size': 33554432,
        'memory_estimate_period': 1000000,
        'grace_period': 200,
        'split_criterion': 'info_gain',
        'split_confidence': .0000001,
        'tie_threshold': .05,
        'binary_split': False,
        'stop_mem_management': False,
        'remove_poor_atts': False,
        'no_preprune': False,
        'leaf_prediction': 'nba',
        'nb_threshold': 0,
        'nominal_attributes': []  # <-- if this is empty assume all attributes are numerical
    }


def classification_incremental_sgd_classifier():
    return {
        'loss': 'hinge',
        'penalty': 'l2',
        'alpha': 0.0001,
        'l1_ratio': 0.15,
        'fit_intercept': True,
        'max_iter': None,
        'tol': 1e-3,
        'eta0': 0.0,
        'power_t': 0.5,
        'early_stopping': False,
        'n_iter_no_change': 5,
        'validation_fraction': 0.1,
    }


def classification_incremental_perceptron():
    return {
        'penalty': None,
        'alpha': 0.0001,
        'fit_intercept': True,
        'max_iter': None,
        'tol': 1e-3,
        'shuffle': True,
        'eta0': 1,
        'early_stopping': False,
        'validation_fraction': 0.1,
        'n_iter_no_change': 5
    }


def classification_xgboost():
    return {
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100
    }


def classification_nn():
    return {
        'n_hidden_layers': 1,
        'n_hidden_units': 10,
        'activation': 'sigmoid',
        'n_epochs': 10,
        'dropout_rate': 0.0
    }


def regression_random_forest():
    return {
        'n_estimators': 10,
        'max_depth': None,
        'max_features': 'auto',
        'n_jobs': -1,
        'random_state': 21
    }


def regression_lasso():
    return {
        'alpha': 1.0,
        'fit_intercept': True,
        'normalize': False,
        'random_state': 21
    }


def regression_linear():
    return {
        'fit_intercept': True,
        'n_jobs': -1,
        'normalize': False
    }


def regression_xgboost():
    return {
        'n_estimators': 100,
        'max_depth': 3
    }


def regression_nn():
    return {
        'n_hidden_layers': 1,
        'n_hidden_units': 10,
        'activation': 'sigmoid',
        'n_epochs': 10,
        'dropout_rate': 0.0
    }


def time_series_prediction_rnn():
    return {
        'n_units': 16,
        'rnn_type': 'lstm',
        'n_epochs': 10,
    }


def _update_incremental_naive_bayes():
    return {
        'alpha': 1.0,
        'fit_prior': True,
        'class_prior': None
    }


def _update_incremental_adaptive_tree():
    return {
        'max_byte_size': 33554432,
        'memory_estimate_period': 1000000,
        'grace_period': 3,
        'split_criterion': 'info_gain',
        'split_confidence': .0000007,
        'tie_threshold': .35,
        'binary_split': False,
        'stop_mem_management': False,
        'remove_poor_atts': False,
        'no_preprune': False,
        'leaf_prediction': 'nba',
        'nb_threshold': 0.4,
        'nominal_attributes': []  # <-- TODO: if this is empty assume all attributes are numerical
    }


def _update_incremental_hoeffding_tree():
    return {
        'max_byte_size': 33554432,
        'memory_estimate_period': 1000000,
        'grace_period': 6,
        'split_criterion': 'gini',
        'split_confidence': .0000008,
        'tie_threshold': .6,
        'binary_split': False,
        'stop_mem_management': False,
        'remove_poor_atts': True,
        'no_preprune': False,
        'leaf_prediction': 'mc',
        'nb_threshold': 0.3,
        'nominal_attributes': []  # <-- TODO: if this is empty assume all attributes are numerical
    }


# Map method config to a dict
CONF_MAP = {
    CLASSIFICATION_RANDOM_FOREST: classification_random_forest,
    CLASSIFICATION_KNN: classification_knn,
    CLASSIFICATION_DECISION_TREE: classification_decision_tree,
    CLASSIFICATION_XGBOOST: classification_xgboost,
    CLASSIFICATION_MULTINOMIAL_NAIVE_BAYES: classification_incremental_naive_bayes,
    CLASSIFICATION_ADAPTIVE_TREE: classification_incremental_adaptive_tree,
    CLASSIFICATION_HOEFFDING_TREE: classification_incremental_hoeffding_tree,
    CLASSIFICATION_SGDC: classification_incremental_sgd_classifier,
    CLASSIFICATION_PERCEPTRON: classification_incremental_perceptron,
    CLASSIFICATION_NN: classification_nn,

    REGRESSION_RANDOM_FOREST: regression_random_forest,
    REGRESSION_XGBOOST: regression_xgboost,
    REGRESSION_LASSO: regression_lasso,
    REGRESSION_LINEAR: regression_linear,
    REGRESSION_NN: regression_nn,

    TIME_SERIES_PREDICTION_RNN: time_series_prediction_rnn,

    UPDATE_INCREMENTAL_NAIVE_BAYES: _update_incremental_naive_bayes,
    UPDATE_INCREMENTAL_ADAPTIVE_TREE: _update_incremental_adaptive_tree,
    UPDATE_INCREMENTAL_HOEFFDING_TREE: _update_incremental_hoeffding_tree
}
