from enum import Enum

from django.db import models

from src.common.models import CommonModel


class PredictiveModelTypes(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    TIME_SERIES_PREDICTION = 'timeSeriesPrediction'


PREDICTIVE_MODEL_TYPE_MAPPINGS = (
    (PredictiveModelTypes.CLASSIFICATION.value, 'classification'),
    (PredictiveModelTypes.REGRESSION.value, 'regression'),
    (PredictiveModelTypes.TIME_SERIES_PREDICTION.value, 'timeSeriesPrediction')
)


class PredictiveModel(CommonModel):
    """Container of Classification to be shown in frontend"""
    model_path = models.FilePathField(path='cache/model_cache/')
    type = models.CharField(choices=PREDICTIVE_MODEL_TYPE_MAPPINGS, default='uniform', max_length=20)

    @staticmethod
    def init(prediction_type: str = PredictiveModelTypes.CLASSIFICATION.value, configuration: dict = None):
        if prediction_type == PredictiveModelTypes.CLASSIFICATION.value:
            from src.predictive_model.classification.models import Classification
            return Classification.init(configuration)
        elif prediction_type == PredictiveModelTypes.REGRESSION.value:
            from src.predictive_model.regression.models import Regression
            return Regression.init(configuration)
        elif prediction_type == PredictiveModelTypes.TIME_SERIES_PREDICTION.value:
            from src.predictive_model.time_series_prediction.models import TimeSeriesPrediction
            return TimeSeriesPrediction.init(configuration)
        else:
            raise ValueError('predictive model type ' + prediction_type + ' not recognized')
