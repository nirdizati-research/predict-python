from django.db import models

from src.clustering.models import Clustering
from src.encoding.models import Encoding
from src.labelling.models import Labelling
from src.logs.models import Log
from src.predictive_model.models import PredictiveModel
from src.split.models import Split


class Cache(models.Model):

    def to_dict(self) -> dict:
        return {}


class LoadedLog(Cache):
    train_log = models.FilePathField(path='cache/labeled_log_cache/')
    test_log = models.FilePathField(path='cache/labeled_log_cache/')

    def to_dict(self) -> dict:
        return {}


class LabelledLogs(LoadedLog):
    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)
    encoding = models.ForeignKey(Encoding, on_delete=models.DO_NOTHING, null=True)
    labelling = models.ForeignKey(Labelling, on_delete=models.DO_NOTHING, null=True)

    def to_dict(self) -> dict:
        return {}
