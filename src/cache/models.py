from django.db import models

from src.clustering.models import Clustering
from src.common.models import CommonModel
from src.encoding.models import Encoding
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel
from src.split.models import Split


class Cache(CommonModel):
    pass


class LoadedLog(Cache):
    train_log_path = models.FilePathField(path='cache/loaded_log_cache/')
    test_log_path = models.FilePathField(path='cache/loaded_log_cache/')
    additional_columns_path = models.FilePathField(path='cache/loaded_log_cache/', null=True)
    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)


class LabelledLog(Cache):
    train_log_path = models.FilePathField(path='cache/labeled_log_cache/')
    test_log_path = models.FilePathField(path='cache/labeled_log_cache/')
    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)
    encoding = models.ForeignKey(Encoding, on_delete=models.DO_NOTHING, null=True)
    labelling = models.ForeignKey(Labelling, on_delete=models.DO_NOTHING, null=True)
