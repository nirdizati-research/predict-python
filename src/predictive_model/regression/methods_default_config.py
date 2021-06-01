def regression_random_forest():
    """"
    Returns the default parameters for regression random forest
    """
    return {
        'n_estimators': 10,
        'max_depth': None,
        'max_features': 'auto',
        'n_jobs': -1,
        'random_state': 21
    }


def regression_lasso():
    """"
    Returns the default parameters for regression lasso
    """
    return {
        'alpha': 1.0,
        'fit_intercept': True,
        'normalize': False,
        'random_state': 21
    }


def regression_linear():
    """"
    Returns the default parameters for regression linear
    """
    return {
        'fit_intercept': True,
        'n_jobs': -1,
        'normalize': False
    }


def regression_xgboost():
    """"
    Returns the default parameters for regression XGBoost
    """
    return {
        'n_estimators': 100,
        'max_depth': 3
    }


def regression_nn():
    """"
    Returns the default parameters for regression NN
    """
    return {
        'n_hidden_layers': 1,
        'n_hidden_units': 10,
        'activation': 'sigmoid',
        'n_epochs': 10,
        'dropout_rate': 0.0
    }
