from django.db import models

from src.predictive_model.models import PredictiveModelBase


class Regression(PredictiveModelBase):
    """Container of Regression to be shown in frontend"""
    clustering = models.ForeignKey('clustering.Clustering', on_delete=models.DO_NOTHING, blank=True, null=True)
    config = models.ForeignKey('RegressorBase', on_delete=models.DO_NOTHING, blank=True, null=True)

    def to_dict(self):
        return {
            'clustering': self.clustering,
            'config': self.config
        }


class RegressorBase(models.Model):
    def to_dict(self):
        return {}


class RandomForest(RegressorBase):
    n_estimators = models.PositiveIntegerField()
    max_features = models.FloatField()
    max_depth = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'max_depth': self.max_depth
        }


class Lasso(RegressorBase):
    alpha = models.FloatField()
    fit_intercept = models.BooleanField()
    normalize = models.BooleanField()

    def to_dict(self):
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'normalize': self.normalize
        }


class Linear(RegressorBase):
    fit_intercept = models.BooleanField()
    normalize = models.BooleanField()

    def to_dict(self):
        return {
            'fit_intercept': self.fit_intercept,
            'normalize': self.normalize
        }


class XGBoost(RegressorBase):
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


class NeuralNetworks(RegressorBase):
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
