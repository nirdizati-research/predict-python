from enum import Enum

from django.db import models

from src.common.models import CommonModel


class PredictionTypes(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    TIME_SERIES_PREDICTION = 'timeSeriesPrediction'


PREDICTION_TYPE_MAPPINGS = (
    (PredictionTypes.CLASSIFICATION.value, 'classification'),
    (PredictionTypes.REGRESSION.value, 'regression'),
    (PredictionTypes.TIME_SERIES_PREDICTION.value, 'timeSeriesPrediction')
)


class PredictiveModel(CommonModel):
    """Container of Classification to be shown in frontend"""
    model_path = models.FilePathField(path='cache/model_cache/')
    predictive_model = models.CharField(choices=PREDICTION_TYPE_MAPPINGS, default='classification', max_length=20)
    prediction_method = models.CharField(max_length=20)

    # noinspection PyDefaultArgument
    @staticmethod
    def init(configuration: dict = None):
        prediction_type = configuration['predictive_model']
        if prediction_type == PredictionTypes.CLASSIFICATION.value:
            from src.predictive_model.classification.models import Classification
            return Classification.init(configuration)
        elif prediction_type == PredictionTypes.REGRESSION.value:
            from src.predictive_model.regression.models import Regression
            return Regression.init(configuration)
        elif prediction_type == PredictionTypes.TIME_SERIES_PREDICTION.value:
            from src.predictive_model.time_series_prediction.models import TimeSeriesPrediction
            return TimeSeriesPrediction.init(configuration)
        else:
            raise ValueError('predictive model type ' + prediction_type + ' not recognized')

    def to_dict(self):
        return {
            'model_path': self.model_path,
            'predictive_model': self.predictive_model
        }
