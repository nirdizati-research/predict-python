from enum import Enum

from django.db import models

from src.clustering.models import Clustering
from src.encoding.models import Encoding
from src.evaluation.models import Evaluation
from src.hyperparameter_optimization.models import HyperparameterOptimization
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel
from src.split.models import Split


class JobStatuses(Enum):
    CREATED = 'created'
    COMPLETED = 'completed'
    ERROR = 'error'
    RUNNING = 'running'


class JobTypes(Enum):
    PREDICTION = 'prediction'
    LABELLING = 'labelling'
    UPDATE = 'update'


JOB_STATUS_MAPPINGS = (
    (JobStatuses.CREATED.value, 'created'),
    (JobStatuses.COMPLETED.value, 'completed'),
    (JobStatuses.ERROR.value, 'error'),
    (JobStatuses.RUNNING.value, 'running')
)

JOB_TYPE_MAPPINGS = (
    (JobTypes.PREDICTION.value, 'prediction'),
    (JobTypes.LABELLING.value, 'labelling'),
    (JobTypes.UPDATE.value, 'update')
)


class Job(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)

    error = models.CharField(default='', max_length=200)
    status = models.CharField(choices=JOB_STATUS_MAPPINGS, default=JobStatuses.CREATED.value, max_length=20)
    type = models.CharField(choices=JOB_TYPE_MAPPINGS, default=JobTypes.PREDICTION.value, max_length=20)

    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)
    encoding = models.ForeignKey(Encoding, on_delete=models.DO_NOTHING, null=True)
    labelling = models.ForeignKey(Labelling, on_delete=models.DO_NOTHING, null=True)
    clustering = models.ForeignKey(Clustering, on_delete=models.DO_NOTHING, null=True)
    predictive_model = models.ForeignKey(PredictiveModel, on_delete=models.DO_NOTHING, null=True)
    evaluation = models.ForeignKey(Evaluation, on_delete=models.DO_NOTHING, null=True)
    hyperparameter_optimizer = models.ForeignKey(HyperparameterOptimization, on_delete=models.DO_NOTHING, null=True)

    @staticmethod
    def to_dict() -> dict:
        return {
            # **self.config,
            # 'type': self.type,
            # 'split': self.split.to_dict(),
            # 'label': LabelContainer(**self.config['label']),
            # 'encoding': EncodingContainer(**self.config['encoding'])
        }
