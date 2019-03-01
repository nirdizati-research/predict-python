from django.db import models


class Evaluation(models.Model):
    """Container of Classification to be shown in frontend"""
    metrics = models.ForeignKey('Metrics', on_delete=models.DO_NOTHING, blank=True, null=True)

    def to_dict(self) -> dict:
        return {
            'metrics': self.metrics
        }


class Metrics(models.Model):
    elapsed_time = models.FloatField()

    def to_dict(self) -> dict:
        return {
            'elapsed_time': self.elapsed_time
        }


class ClassificationMetrics(Metrics):
    f1_score = models.FloatField()
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()

    def to_dict(self) -> dict:
        return {
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall
        }


class BinaryClassificationMetrics(ClassificationMetrics):
    true_positive = models.FloatField()
    true_negative = models.FloatField()
    false_negative = models.FloatField()
    false_positive = models.FloatField()
    auc = models.FloatField()

    def to_dict(self) -> dict:
        return {
            'true_positive': self.true_positive,
            'true_negative': self.true_negative,
            'false_negative': self.false_negative,
            'false_positive': self.false_positive,
            'auc': self.auc
        }


class MulticlassClassificationMetrics(ClassificationMetrics):
    def to_dict(self) -> dict:
        return {}


class RegressionMetrics(Metrics):
    rmse = models.FloatField()
    mae = models.FloatField()
    mape = models.FloatField()

    def to_dict(self) -> dict:
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape
        }


class TimeSeriesPredictionMetrics(Metrics):
    def to_dict(self) -> dict:
        return {}
