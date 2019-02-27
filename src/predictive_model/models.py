from django.db import models


class PredictiveModel(models.Model):
    """Container of Classification to be shown in frontend"""
    split = models.ForeignKey('split.Split', on_delete=models.DO_NOTHING, blank=True, null=True)
    encoding = models.ForeignKey('encoding.Encoding', on_delete=models.DO_NOTHING, blank=True, null=True)
    labelling = models.ForeignKey('labelling.Labelling', on_delete=models.DO_NOTHING, blank=True, null=True)
    model_type = models.ForeignKey('PredictiveModelBase', on_delete=models.DO_NOTHING, blank=True, null=True)

    def to_dict(self):
        return {
            'split': self.split,
            'encoding': self.encoding,
            'labelling': self.labelling,
            'model_type': self.model_type
        }


class PredictiveModelBase(models.Model):
    def to_dict(self):
        return {}
