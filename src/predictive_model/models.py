from enum import Enum

from django.db import models


class PredictiveModelTypes(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    TIME_SERIES_PREDICTION = 'timeSeriesPrediction'


class PredictiveModel(models.Model):
    """Container of Classification to be shown in frontend"""

    @staticmethod
    def init(prediction_type: str = PredictiveModelTypes.CLASSIFICATION, configuration: dict = None):
        if prediction_type == PredictiveModelTypes.CLASSIFICATION:
            from src.predictive_model.classification.models import Classification
            return Classification.init(configuration)
        elif prediction_type == PredictiveModelTypes.REGRESSION:
            from src.predictive_model.regression.models import Regression
            return Regression.init(configuration)
        elif prediction_type == PredictiveModelTypes.TIME_SERIES_PREDICTION:
            from src.predictive_model.time_series_prediction.models import TimeSeriesPrediction
            return TimeSeriesPrediction.init(configuration)
        else:
            raise ValueError('predictive model type ' + prediction_type + ' not recognized')

    def to_dict(self):
        return {}
