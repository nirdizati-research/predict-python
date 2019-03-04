from enum import Enum

from django.db import models

from src.predictive_model.models import PredictiveModel, PredictionTypes
from src.predictive_model.regression.methods_default_config import regression_random_forest, regression_lasso, \
    regression_linear, regression_xgboost, regression_nn


class RegressionMethods(Enum):
    LINEAR = 'linear'
    RANDOM_FOREST = 'randomForest'
    LASSO = 'lasso'
    XGBOOST = 'xgboost'
    NN = 'nn'


REGRESSION_LASSO = '{}.{}'.format(PredictionTypes.REGRESSION.value, RegressionMethods.LASSO.value)
REGRESSION_LINEAR = '{}.{}'.format(PredictionTypes.REGRESSION.value, RegressionMethods.LINEAR.value)
REGRESSION_XGBOOST = '{}.{}'.format(PredictionTypes.REGRESSION.value, RegressionMethods.XGBOOST.value)
REGRESSION_RANDOM_FOREST = '{}.{}'.format(PredictionTypes.REGRESSION.value, RegressionMethods.RANDOM_FOREST.value)
REGRESSION_NN = '{}.{}'.format(PredictionTypes.REGRESSION.value, RegressionMethods.NN.value)

REGRESSION_METHODS = (
    (RegressionMethods.LINEAR.value, 'linear'),
    (RegressionMethods.RANDOM_FOREST.value, 'randomForest'),
    (RegressionMethods.LASSO.value, 'lasso'),
    (RegressionMethods.XGBOOST.value, 'xgboost'),
    (RegressionMethods.NN.value, 'nn')
)


class Regression(PredictiveModel):
    """Container of Regression to be shown in frontend"""
    predictive_model_type = PredictionTypes.REGRESSION.value

    # noinspection PyDefaultArgument
    @staticmethod
    def init(configuration: dict = {'type': RegressionMethods.RANDOM_FOREST.value}):
        regressor_type = configuration['type']
        if regressor_type == RegressionMethods.RANDOM_FOREST.value:
            default_configuration = regression_random_forest()
            return RandomForest.objects.get_or_create(
                prediction_method=RegressionMethods.RANDOM_FOREST.value,
                n_estimators=configuration.get('n_estimators', default_configuration['n_estimators']),
                max_features=configuration.get('max_features', default_configuration['max_features']),
                max_depth=configuration.get('max_depth', default_configuration['max_depth'])
            )
        elif regressor_type == RegressionMethods.LASSO.value:
            default_configuration = regression_lasso()
            return Lasso.objects.get_or_create(
                prediction_method=RegressionMethods.LASSO.value,
                alpha=configuration.get('alpha', default_configuration['alpha']),
                fit_intercept=configuration.get('fit_intercept', default_configuration['fit_intercept']),
                normalize=configuration.get('normalize', default_configuration['normalize'])
            )
        elif regressor_type == RegressionMethods.LINEAR.value:
            default_configuration = regression_linear()
            return Linear.objects.get_or_create(
                prediction_method=RegressionMethods.LINEAR.value,
                fit_intercept=configuration.get('fit_intercept', default_configuration['fit_intercept']),
                normalize=configuration.get('normalize', default_configuration['normalize']),
            )
        elif regressor_type == RegressionMethods.XGBOOST.value:
            default_configuration = regression_xgboost()
            return XGBoost.objects.get_or_create(
                prediction_method=RegressionMethods.XGBOOST.value,
                max_depth=configuration.get('max_depth', default_configuration['max_depth']),
                n_estimators=configuration.get('n_estimators', default_configuration['n_estimators'])
            )
        elif regressor_type == RegressionMethods.NN.value:
            default_configuration = regression_nn()
            return NeuralNetwork.objects.get_or_create(
                prediction_method=RegressionMethods.NN.value,
                hidden_layers=configuration.get('hidden_layers', default_configuration['hidden_layers']),
                hidden_units=configuration.get('hidden_units', default_configuration['hidden_units']),
                activation_function=configuration.get('activation_function',
                                                      default_configuration['activation_function']),
                epochs=configuration.get('epochs', default_configuration['epochs']),
                dropout_rate=configuration.get('dropout_rate', default_configuration['dropout_rate'])
            )
        else:
            raise ValueError('regressor type ' + regressor_type + ' not recognized')


class RandomForest(Regression):
    n_estimators = models.PositiveIntegerField()
    max_features = models.FloatField()
    max_depth = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'max_depth': self.max_depth
        }


class Lasso(Regression):
    alpha = models.FloatField()
    fit_intercept = models.BooleanField()
    normalize = models.BooleanField()

    def to_dict(self):
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'normalize': self.normalize
        }


class Linear(Regression):
    fit_intercept = models.BooleanField()
    normalize = models.BooleanField()

    def to_dict(self):
        return {
            'fit_intercept': self.fit_intercept,
            'normalize': self.normalize
        }


class XGBoost(Regression):
    max_depth = models.PositiveIntegerField()
    n_estimators = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators
        }


NEURAL_NETWORKS_ACTIVATION_FUNCTION = (
    ('sigmoid', 'sigmoid'),
    ('tanh', 'tanh'),
    ('relu', 'relu')
)


class NeuralNetwork(Regression):
    hidden_layers = models.PositiveIntegerField()
    hidden_units = models.PositiveIntegerField()
    activation_function = models.CharField(choices=NEURAL_NETWORKS_ACTIVATION_FUNCTION, default='relu',
                                           max_length=20)
    epochs = models.PositiveIntegerField()
    dropout_rate = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'hidden_layers': self.hidden_layers,
            'hidden_units': self.hidden_units,
            'activation_function': self.activation_function,
            'epochs': self.epochs,
            'dropout_rate': self.dropout_rate

        }
