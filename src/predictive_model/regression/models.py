from enum import Enum

from django.db import models

from src.predictive_model.models import PredictiveModel, PredictiveModels
from src.predictive_model.regression.methods_default_config import regression_random_forest, regression_lasso, \
    regression_linear, regression_xgboost, regression_nn


class RegressionMethods(Enum):
    LINEAR = 'linear'
    RANDOM_FOREST = 'randomForest'
    LASSO = 'lasso'
    XGBOOST = 'xgboost'
    NN = 'nn'


REGRESSION_LASSO = '{}.{}'.format(PredictiveModels.REGRESSION.value, RegressionMethods.LASSO.value)
REGRESSION_LINEAR = '{}.{}'.format(PredictiveModels.REGRESSION.value, RegressionMethods.LINEAR.value)
REGRESSION_XGBOOST = '{}.{}'.format(PredictiveModels.REGRESSION.value, RegressionMethods.XGBOOST.value)
REGRESSION_RANDOM_FOREST = '{}.{}'.format(PredictiveModels.REGRESSION.value, RegressionMethods.RANDOM_FOREST.value)
REGRESSION_NN = '{}.{}'.format(PredictiveModels.REGRESSION.value, RegressionMethods.NN.value)

REGRESSION_METHOD_MAPPINGS = (
    (RegressionMethods.LINEAR.value, 'linear'),
    (RegressionMethods.RANDOM_FOREST.value, 'randomForest'),
    (RegressionMethods.LASSO.value, 'lasso'),
    (RegressionMethods.XGBOOST.value, 'xgboost'),
    (RegressionMethods.NN.value, 'nn')
)


class Regression(PredictiveModel):
    """Container of Regression to be shown in frontend"""

    @staticmethod
    def init(configuration: dict):
        regressor_type = configuration['prediction_method']
        if regressor_type == RegressionMethods.RANDOM_FOREST.value:
            default_configuration = regression_random_forest()
            return RandomForest.objects.create(
                prediction_method=regressor_type,
                predictive_model=PredictiveModels.REGRESSION.value,
                n_estimators=configuration.get('n_estimators', default_configuration['n_estimators']),
                max_features=configuration.get('max_features', default_configuration['max_features']),
                max_depth=configuration.get('max_depth', default_configuration['max_depth']),
            )
        elif regressor_type == RegressionMethods.LASSO.value:
            default_configuration = regression_lasso()
            return Lasso.objects.create(
                prediction_method=regressor_type,
                predictive_model=PredictiveModels.REGRESSION.value,
                alpha=configuration.get('alpha', default_configuration['alpha']),
                fit_intercept=configuration.get('fit_intercept', default_configuration['fit_intercept']),
                normalize=configuration.get('normalize', default_configuration['normalize'])
            )
        elif regressor_type == RegressionMethods.LINEAR.value:
            default_configuration = regression_linear()
            return Linear.objects.create(
                prediction_method=regressor_type,
                predictive_model=PredictiveModels.REGRESSION.value,
                fit_intercept=configuration.get('fit_intercept', default_configuration['fit_intercept']),
                normalize=configuration.get('normalize', default_configuration['normalize']),
            )
        elif regressor_type == RegressionMethods.XGBOOST.value:
            default_configuration = regression_xgboost()
            return XGBoost.objects.create(
                prediction_method=regressor_type,
                predictive_model=PredictiveModels.REGRESSION.value,
                max_depth=configuration.get('max_depth', default_configuration['max_depth']),
                n_estimators=configuration.get('n_estimators', default_configuration['n_estimators'])
            )
        elif regressor_type == RegressionMethods.NN.value:
            default_configuration = regression_nn()
            return NeuralNetwork.objects.create(
                prediction_method=regressor_type,
                predictive_model=PredictiveModels.REGRESSION.value,
                n_hidden_layers=configuration.get('n_hidden_layers', default_configuration['n_hidden_layers']),
                n_hidden_units=configuration.get('n_hidden_units', default_configuration['n_hidden_units']),
                activation=configuration.get('activation', default_configuration['activation']),
                n_epochs=configuration.get('n_epochs', default_configuration['n_epochs']),
                dropout_rate=configuration.get('dropout_rate', default_configuration['dropout_rate'])
            )
        else:
            raise ValueError('regressor type {} not recognized'.format(regressor_type))


RANDOM_FOREST_MAX_FEATURES_MAPPINGS = (
    ('auto', 'auto'),
    ('sqrt', 'sqrt'),
    ('log2', 'log2')
)


class RandomForest(Regression):
    n_estimators = models.PositiveIntegerField()
    max_features = models.CharField(choices=RANDOM_FOREST_MAX_FEATURES_MAPPINGS, null=True, default=None, max_length=max(len(el[1]) for el in RANDOM_FOREST_MAX_FEATURES_MAPPINGS) + 1)
    max_depth = models.PositiveIntegerField(null=True)
    random_state = 21

    def to_dict(self):
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'max_depth': self.max_depth,
            'random_state': self.random_state
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


NEURAL_NETWORKS_ACTIVATION = (
    ('sigmoid', 'sigmoid'),
    ('tanh', 'tanh'),
    ('relu', 'relu')
)


class NeuralNetwork(Regression):
    n_hidden_layers = models.PositiveIntegerField()
    n_hidden_units = models.PositiveIntegerField()
    activation = models.CharField(choices=NEURAL_NETWORKS_ACTIVATION, default='relu', max_length=max(len(el[1]) for el in NEURAL_NETWORKS_ACTIVATION) + 1)
    n_epochs = models.PositiveIntegerField()
    dropout_rate = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'n_hidden_layers': self.n_hidden_layers,
            'n_hidden_units': self.n_hidden_units,
            'activation': self.activation,
            'n_epochs': self.n_epochs,
            'dropout_rate': self.dropout_rate

        }
