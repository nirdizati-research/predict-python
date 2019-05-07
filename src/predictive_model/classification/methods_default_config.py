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
        'n_neighbors': 3,
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
        'epsilon': 0.1,
        'learning_rate': 'optimal',
        'average': False
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
