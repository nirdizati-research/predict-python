from django.db import models

from src.common.models import CommonModel
from src.predictive_model.models import PredictiveModels


class Evaluation(CommonModel):
    @staticmethod
    def init(prediction_type, results):
        if prediction_type == PredictiveModels.CLASSIFICATION.value:
            pass #TODO fixme
            # if labels == BINARY:
            #     BinaryClassificationMetrics.objects.get_or_create(
            #         metrics='',
            #
            #         true_positive=results['true_positive'],
            #         true_negative=results['true_negative'],
            #         false_negative=results['false_negative'],
            #         false_positive=results['false_positive'],
            #         auc=results['auc']
            #     )
            # elif labels == MUTLICLASS:
            #     MulticlassClassificationMetrics.objects.get_or_create(metrics='')
        elif prediction_type == PredictiveModels.REGRESSION.value:
            return RegressionMetrics.objects.get_or_create(
                rmse=results['rmse'],
                rscore=results['rscore'],
                mae=results['mae'],
                mape=results['mape']
            )[0]
        elif prediction_type == PredictiveModels.TIME_SERIES_PREDICTION.value:
            TimeSeriesPredictionMetrics.objects.get_or_create()


class ClassificationMetrics(Evaluation):
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


class BinaryClassificationMetrics(Evaluation):
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
    pass


class RegressionMetrics(Evaluation):
    rmse = models.FloatField()
    mae = models.FloatField()
    rscore = models.FloatField()
    mape = models.FloatField()

    def to_dict(self) -> dict:
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'rscore': self.rscore
        }


class TimeSeriesPredictionMetrics(Evaluation):
    pass
