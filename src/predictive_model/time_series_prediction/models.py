from enum import Enum

from django.db import models

from src.core.default_configuration import time_series_prediction_rnn
from src.predictive_model.models import PredictiveModel


class TimeSeriesPredictionMethods(Enum):
    RNN = 'rnn'

class TimeSeriesPrediction(PredictiveModel):
    """Container of Classification to be shown in frontend"""

    @staticmethod
    def init(configuration: dict = {'type': TimeSeriesPredictionMethods.RNN}):
        time_series_predictor_type = configuration['type']
        if time_series_predictor_type == TimeSeriesPredictionMethods.RNN:
            default_configuration = time_series_prediction_rnn()
            return RecurrentNeuralNetwork.objects.get_or_create(
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
    n_units = models.PositiveIntegerField()
    rnn_type = models.CharField(choices=RNN_TYPES, default='lstm', max_length=10)
    n_epochs = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'n_units': self.n_units,
            'rnn_type': self.rnn_type,
            'n_epochs': self.n_epochs
        }
