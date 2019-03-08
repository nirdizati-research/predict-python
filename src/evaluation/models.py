from django.db import models

from src.common.models import CommonModel
from src.predictive_model.models import PredictiveModels


class Evaluation(CommonModel):
    @staticmethod
    def init(prediction_type, results, binary=False):
        if prediction_type == PredictiveModels.CLASSIFICATION.value:
            if binary:
                return BinaryClassificationMetrics.objects.get_or_create(
                    true_positive=results['true_positive'],
                    true_negative=results['true_negative'],
                    false_negative=results['false_negative'],
                    false_positive=results['false_positive'],
                    auc=results['auc']
                )[0]
            else:
                return MulticlassClassificationMetrics.objects.get_or_create(
                )[0]
        elif prediction_type == PredictiveModels.REGRESSION.value:
            return RegressionMetrics.objects.get_or_create(
                rmse=results['rmse'],
                rscore=results['rscore'],
                mae=results['mae'],
                mape=results['mape']
            )[0]
        elif prediction_type == PredictiveModels.TIME_SERIES_PREDICTION.value:
            return TimeSeriesPredictionMetrics.objects.get_or_create(
            )[0]
        else:
            raise ValueError('evaluation model type ' + prediction_type + ' not recognized')


class ClassificationMetrics(Evaluation):
    f1_score = models.FloatField(blank=True, null=True)
    accuracy = models.FloatField(blank=True, null=True)
    precision = models.FloatField(blank=True, null=True)
    recall = models.FloatField(blank=True, null=True)

    def to_dict(self) -> dict:
        return {
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall
        }


class BinaryClassificationMetrics(Evaluation):
    true_positive = models.FloatField(blank=True, null=True)
    true_negative = models.FloatField(blank=True, null=True)
    false_negative = models.FloatField(blank=True, null=True)
    false_positive = models.FloatField(blank=True, null=True)
    auc = models.FloatField(blank=True, null=True)

    def to_dict(self) -> dict:
        return {
            'true_positive': self.true_positive,
            'true_negative': self.true_negative,
            'false_negative': self.false_negative,
            'false_positive': self.false_positive,
            'auc': self.auc
        }


class MulticlassClassificationMetrics(ClassificationMetrics):
    pass


class RegressionMetrics(Evaluation):
    rmse = models.FloatField(blank=True, null=True)
    mae = models.FloatField(blank=True, null=True)
    rscore = models.FloatField(blank=True, null=True)
    mape = models.FloatField(blank=True, null=True)

    def to_dict(self) -> dict:
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'rscore': self.rscore
        }


class TimeSeriesPredictionMetrics(Evaluation):
    pass
