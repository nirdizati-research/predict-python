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
