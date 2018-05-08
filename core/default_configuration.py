# Default configurations
from core.constants import *


def _classification_random_forest():
    return {
        'n_estimators': 10,
        'max_depth': None,
        'max_features': 'auto',
        'n_jobs': -1,
        'random_state': 21
    }


def _classification_knn():
    return {
        'n_neighbors': 5,
        'n_jobs': -1,
        'weights': 'uniform'
    }


def _classification_decision_tree():
    return {
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 21
    }


def _regression_random_forest():
    return {
        'n_estimators': 10,
        'max_depth': None,
        'max_features': 'auto',
        'n_jobs': -1,
        'random_state': 21
    }


def _regression_lasso():
    return {
        'alpha': 1.0,
        'fit_intercept': True,
        'normalize': False,
        'random_state': 21
    }


def _regression_linear():
    return {
        'fit_intercept': True,
        'n_jobs': -1,
        'normalize': False
    }


def _kmeans():
    return {
        'n_clusters': 3,
        'max_iter': 300,
        'n_jobs': -1,
        'algorithm': 'auto',
        'random_state': 21
    }


# Map method config to a dict
CONF_MAP = {CLASSIFICATION_RANDOM_FOREST: _classification_random_forest, CLASSIFICATION_KNN: _classification_knn,
            CLASSIFICATION_DECISION_TREE: _classification_decision_tree,
            REGRESSION_RANDOM_FOREST: _regression_random_forest,
            REGRESSION_LASSO: _regression_lasso, REGRESSION_LINEAR: _regression_linear}
