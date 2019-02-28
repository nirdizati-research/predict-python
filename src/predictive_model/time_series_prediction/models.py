from django.db import models

from src.predictive_model.models import PredictiveModel


class TimeSeriesPrediction(PredictiveModel):
    """Container of Classification to be shown in frontend"""

    def to_dict(self):
        return {}


RNN_TYPES = (
    ('lstm', 'lstm'),
    ('gru', 'gru')
)


class RNN(TimeSeriesPrediction):
    n_units = models.PositiveIntegerField()
    rnn_type = models.CharField(choices=RNN_TYPES, default='lstm', max_length=10)
    n_epochs = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'n_units': self.n_units,
            'rnn_type': self.rnn_type,
            'n_epochs': self.n_epochs
        }
