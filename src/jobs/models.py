from django.db import models

from pred_models.models import TYPES
from src.clustering.models import Clustering
from src.evaluation.models import Evaluation
from src.labelling.models import Labelling
from src.encoding.models import Encoding
from src.encoding.encoding_container import EncodingContainer
from src.labelling.label_container import LabelContainer
from src.predictive_model.models import PredictiveModel
from src.split.models import Split

CREATED = 'created'
COMPLETED = 'completed'
ERROR = 'error'
RUNNING = 'running'


STATUSES = (
    (CREATED, 'created'),
    (COMPLETED, 'completed'),
    (ERROR, 'error'),
    (RUNNING, 'running')
)


class Job(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)

    error = models.CharField(default='', max_length=200)
    status = models.CharField(choices=STATUSES, default=CREATED, max_length=20)
    type = models.CharField(choices=TYPES, max_length=20)

    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)
    encoding = models.ForeignKey(Encoding, on_delete=models.DO_NOTHING, null=True)
    labelling = models.ForeignKey(Labelling, on_delete=models.DO_NOTHING, null=True)
    clustering = models.ForeignKey(Clustering, on_delete=models.DO_NOTHING, null=True)
    predictive_model = models.ForeignKey(PredictiveModel, on_delete=models.DO_NOTHING, null=True)
    evaluation = models.ForeignKey(Evaluation, on_delete=models.DO_NOTHING, null=True)

    def to_dict(self) -> dict:
        return {
            # **self.config,
            # 'type': self.type,
            # 'split': self.split.to_dict(),
            # 'label': LabelContainer(**self.config['label']),
            # 'encoding': EncodingContainer(**self.config['encoding'])
        }
