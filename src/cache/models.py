from django.db import models

from src.clustering.models import Clustering
from src.common.models import CommonModel
from src.encoding.models import Encoding
from src.labelling.models import Labelling
from src.logs.models import Log
from src.predictive_model.models import PredictiveModel
from src.split.models import Split


class Cache(CommonModel):
    pass


class LabelledLogs(Cache):
    log = models.ForeignKey(Log, on_delete=models.DO_NOTHING, related_name='base_log', blank=True, null=True)
    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)
    encoding = models.ForeignKey(Encoding, on_delete=models.DO_NOTHING, null=True)
    labelling = models.ForeignKey(Labelling, on_delete=models.DO_NOTHING, null=True)
    clustering = models.ForeignKey(Clustering, on_delete=models.DO_NOTHING, null=True)
    predictive_model = models.ForeignKey(PredictiveModel, on_delete=models.DO_NOTHING, null=True)

    def to_dict(self) -> dict:
        return {}
