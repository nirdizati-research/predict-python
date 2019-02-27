from django.db import models


class Evaluation(models.Model):
    """Container of Classification to be shown in frontend"""
    split = models.ForeignKey('split.Split', on_delete=models.DO_NOTHING, blank=True, null=True)
    encoding = models.ForeignKey('encoding.Encoding', on_delete=models.DO_NOTHING, blank=True, null=True)
    labelling = models.ForeignKey('labelling.Labelling', on_delete=models.DO_NOTHING, blank=True, null=True)
    clustering = models.ForeignKey('clustering.Clustering', on_delete=models.DO_NOTHING, blank=True, null=True)
    predictive_model = models.ForeignKey('predictive_model.PredictiveModelBase', on_delete=models.DO_NOTHING,
                                         blank=True, null=True)
    metrics = models.ForeignKey('MetricsBase', on_delete=models.DO_NOTHING, blank=True, null=True)

    def to_dict(self):
        return {
            'split': self.split,
            'encoding': self.encoding,
            'labelling': self.labelling,
            'clustering': self.clustering,
            'predictive_model': self.predictive_model
        }


class MetricsBase(models.Model):
    elapsed_time = models.FloatField()

    def to_dict(self):
        return {
            'elapsed_time': self.elapsed_time
        }


class ClassificationMetricsBase(MetricsBase):
    f1_score = models.FloatField()
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()

    def to_dict(self):
        return {
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall
        }


class BinaryClassificationMetrics(ClassificationMetricsBase):
    true_positive = models.FloatField()
    true_negative = models.FloatField()
    false_negative = models.FloatField()
    false_positive = models.FloatField()
    auc = models.FloatField()

    def to_dict(self):
        return {
            'true_positive': self.true_positive,
            'true_negative': self.true_negative,
            'false_negative': self.false_negative,
            'false_positive': self.false_positive,
            'auc': self.auc
        }


class MulticlassClassificationMetrics(ClassificationMetricsBase):
    def to_dict(self):
        return {}


class RegressionMetrics(MetricsBase):
    def to_dict(self):
        return {}


class TimeSeriesPredictionMetrics(MetricsBase):
    def to_dict(self):
        return {}
