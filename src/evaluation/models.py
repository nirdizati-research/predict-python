from django.db import models
from model_utils.managers import InheritanceManager

from src.common.models import CommonModel
from src.predictive_model.models import PredictiveModels


class Evaluation(CommonModel):
    elapsed_time = models.DurationField()
    objects = InheritanceManager()

    @staticmethod
    def init(prediction_type, results, binary=False):
        if prediction_type == PredictiveModels.CLASSIFICATION.value:
            if binary:
                return BinaryClassificationMetrics.objects.get_or_create(
                    elapsed_time=results['elapsed_time'] if results['elapsed_time'] != '--' else None,
                    f1_score=results['f1score'] if results['f1score'] != '--' else None,
                    auc=results['auc'] if results['auc'] != '--' else None,
                    accuracy=results['acc'] if results['acc'] != '--' else None,
                    precision=results['precision'] if results['precision'] != '--' else None,
                    recall=results['recall'] if results['recall'] != '--' else None,
                    true_positive=results['true_positive'] if results['true_positive'] != '--' else None,
                    true_negative=results['true_negative'] if results['true_negative'] != '--' else None,
                    false_negative=results['false_negative'] if results['false_negative'] != '--' else None,
                    false_positive=results['false_positive'] if results['false_positive'] != '--' else None,
                )[0]
            else:
                return MulticlassClassificationMetrics.objects.get_or_create(
                    elapsed_time=results['elapsed_time'] if results['elapsed_time'] != '--' else None,
                    f1_score=results['f1score'] if results['f1score'] != '--' else None,
                    accuracy=results['acc'] if results['acc'] != '--' else None,
                    precision=results['precision'] if results['precision'] != '--' else None,
                    recall=results['recall'] if results['recall'] != '--' else None
                )[0]
        elif prediction_type == PredictiveModels.REGRESSION.value:
            return RegressionMetrics.objects.get_or_create(
                elapsed_time=results['elapsed_time'] if results['elapsed_time'] != '--' else None,
                rmse=results['rmse'] if results['rmse'] != '--' else None,
                rscore=results['rscore'] if results['rscore'] != '--' else None,
                mae=results['mae'] if results['mae'] != '--' else None,
                mape=results['mape'] if results['mape'] != '--' else None
            )[0]
        elif prediction_type == PredictiveModels.TIME_SERIES_PREDICTION.value:
            return TimeSeriesPredictionMetrics.objects.get_or_create(
                elapsed_time=results['elapsed_time'] if results['elapsed_time'] != '--' else None,
                nlevenshtein=results['nlevenshtein'] if results['nlevenshtein'] != '--' else None
            )[0]
        else:
            raise ValueError('evaluation model type {} not recognized'.format(prediction_type))

    def to_dict(self) -> dict:
        return{
            'elapsed_time': self.elapsed_time
        }


class ClassificationMetrics(Evaluation):
    f1_score = models.FloatField(blank=True, null=True)
    accuracy = models.FloatField(blank=True, null=True)
    precision = models.FloatField(blank=True, null=True)
    recall = models.FloatField(blank=True, null=True)
    objects = InheritanceManager()

    def to_dict(self) -> dict:
        return {
            'elapsed_time': self.elapsed_time,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall
        }


class BinaryClassificationMetrics(ClassificationMetrics):
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
    nlevenshtein = models.FloatField(blank=True, null=True)

    def to_dict(self) -> dict:
        return {
            'nlevenshtein': self.nlevenshtein
        }
