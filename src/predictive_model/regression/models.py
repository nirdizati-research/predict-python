from django.db import models

from src.predictive_model.models import PredictiveModel


class Regression(PredictiveModel):
    """Container of Regression to be shown in frontend"""

    def to_dict(self):
        return {}


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


class NeuralNetworks(Regression):
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
