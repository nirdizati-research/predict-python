from django.db import models

from src.predictive_model.models import PredictiveModelBase


class TimeSeriesPrediction(PredictiveModelBase):
    """Container of Classification to be shown in frontend"""
    # TODO: shouldnt we add the training data?
    # split = models.ForeignKey('split.Split', on_delete=models.DO_NOTHING, blank=True, null=True)
    # encoding = models.ForeignKey('encoding.Encoding', on_delete=models.DO_NOTHING, blank=True, null=True)
    # labelling = models.ForeignKey('labelling.Labelling', on_delete=models.DO_NOTHING, blank=True, null=True)
    clustering = models.ForeignKey('clustering.Clustering', on_delete=models.DO_NOTHING, blank=True, null=True)
    config = models.ForeignKey('TimeSeriesPredictorBase', on_delete=models.DO_NOTHING, blank=True, null=True)

    def to_dict(self):
        return {
            'clustering': self.clustering,
            'config': self.config
        }


class TimeSeriesPredictorBase(models.Model):
    def to_dict(self):
        return {}


RNN_TYPES = (
    ('lstm', 'lstm'),
    ('gru', 'gru')
)


class RNN(TimeSeriesPredictorBase):
    n_units = models.PositiveIntegerField()
    rnn_type = models.CharField(choices=RNN_TYPES, default='lstm', max_length=10)
    n_epochs = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'n_units': self.n_units,
            'rnn_type': self.rnn_type,
            'n_epochs': self.n_epochs
        }
