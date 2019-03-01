from enum import Enum

from django.db import models

from src.predictive_model.models import PredictiveModel, PredictiveModelTypes
from src.predictive_model.time_series_prediction.methods_default_config import time_series_prediction_rnn


class TimeSeriesPredictionMethods(Enum):
    RNN = 'rnn'


TIME_SERIES_PREDICTION_RNN = '{}.{}'.format(PredictiveModelTypes.TIME_SERIES_PREDICTION.value,
                                            TimeSeriesPredictionMethods.RNN.value)


class TimeSeriesPrediction(PredictiveModel):
    """Container of Classification to be shown in frontend"""

    @staticmethod
    def init(configuration: dict = {'type': TimeSeriesPredictionMethods.RNN.value}):
        time_series_predictor_type = configuration['type']
        if time_series_predictor_type == TimeSeriesPredictionMethods.RNN.value:
            default_configuration = time_series_prediction_rnn()
            return RecurrentNeuralNetwork.objects.get_or_create(
                type=PredictiveModelTypes.TIME_SERIES_PREDICTION.value,
                n_units=configuration.get('n_units', default_configuration['n_units']),
                rnn_type=configuration.get('rnn_type', default_configuration['rnn_type']),
                n_epochs=configuration.get('n_epochs', default_configuration['n_epochs'])
            )
        else:
            raise ValueError('time series predictor type ' + time_series_predictor_type + ' not recognized')

    def to_dict(self):
        return {}


RNN_TYPES = (
    ('lstm', 'lstm'),
    ('gru', 'gru')
)


class RecurrentNeuralNetwork(TimeSeriesPrediction):
    __name__ = TimeSeriesPredictionMethods.RNN.value

    n_units = models.PositiveIntegerField()
    rnn_type = models.CharField(choices=RNN_TYPES, default='lstm', max_length=10)
    n_epochs = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'n_units': self.n_units,
            'rnn_type': self.rnn_type,
            'n_epochs': self.n_epochs
        }
