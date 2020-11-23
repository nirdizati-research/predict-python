from enum import Enum

from src.predictive_model.classification import classification
from src.predictive_model.models import PredictiveModels
from src.predictive_model.regression import regression
from src.predictive_model.time_series_prediction import time_series_prediction


class ModelActions(Enum):
    """ class containing common methods between different ML approaches """
    PREDICT = 'predict'
    PREDICT_PROBA = 'predict_proba'
    UPDATE_AND_TEST = 'update_and_test'
    BUILD_MODEL_AND_TEST = 'build_model_and_test'


MODEL = {
    PredictiveModels.CLASSIFICATION.value: {
        ModelActions.PREDICT.value: classification.predict,
        ModelActions.PREDICT_PROBA.value: classification.predict_proba,
        ModelActions.UPDATE_AND_TEST.value: classification.update_and_test,
        ModelActions.BUILD_MODEL_AND_TEST.value: classification.classification
    },
    PredictiveModels.REGRESSION.value: {
        ModelActions.PREDICT.value: regression.predict,
        ModelActions.BUILD_MODEL_AND_TEST.value: regression.regression
    },
    PredictiveModels.TIME_SERIES_PREDICTION.value: {
        ModelActions.PREDICT.value: time_series_prediction.predict,
        ModelActions.BUILD_MODEL_AND_TEST.value: time_series_prediction.time_series_prediction
    }
}
