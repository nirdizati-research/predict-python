from enum import Enum

from django.db import models

from src.clustering.models import Clustering
from src.common.models import CommonModel
from src.encoding.models import Encoding
from src.evaluation.models import Evaluation
from src.hyperparameter_optimization.models import HyperparameterOptimization
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel, PredictiveModel
from src.split.models import Split


class ModelType(Enum):
    CLUSTERER = 'clusterer'
    CLASSIFIER = 'classification'
    REGRESSOR = 'regression'
    TIME_SERIES_PREDICTOR = 'time_series_prediction'


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


class Job(CommonModel):
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)

    error = models.CharField(default='', max_length=500)
    status = models.CharField(choices=JOB_STATUS_MAPPINGS, default=JobStatuses.CREATED.value, max_length=max(len(el[1]) for el in JOB_STATUS_MAPPINGS)+1)
    type = models.CharField(choices=JOB_TYPE_MAPPINGS, default=JobTypes.PREDICTION.value, max_length=max(len(el[1]) for el in JOB_TYPE_MAPPINGS)+1)
    create_models = models.BooleanField(default=False)

    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)
    encoding = models.ForeignKey(Encoding, on_delete=models.DO_NOTHING, null=True)
    labelling = models.ForeignKey(Labelling, on_delete=models.DO_NOTHING, null=True)
    clustering = models.ForeignKey(Clustering, on_delete=models.DO_NOTHING, null=True)
    predictive_model = models.ForeignKey(PredictiveModel, on_delete=models.DO_NOTHING, null=True)
    evaluation = models.ForeignKey(Evaluation, on_delete=models.DO_NOTHING, null=True)
    hyperparameter_optimizer = models.ForeignKey(HyperparameterOptimization, on_delete=models.DO_NOTHING, null=True)
    incremental_train = models.ForeignKey('self', on_delete=models.DO_NOTHING, related_name='base_model',
                                          null=True)# self-reference

    def to_dict(self) -> dict:
        return {
            'created_date': self.created_date,
            'modified_date': self.modified_date,
            'error': self.error,
            'status': self.status,
            'type': self.type,
            'create_models': self.create_models,
            'split': self.split.to_dict(),
            'encoding': self.encoding.to_dict(),
            'labelling': self.labelling.to_dict(),
            'clustering': self.clustering.to_dict(),
            'predictive_model': self.predictive_model.to_dict(),
            'evaluation': [self.evaluation.to_dict() if self.evaluation is not None else None],
            'hyperparameter_optimizer': [
                self.hyperparameter_optimizer.to_dict() if self.hyperparameter_optimizer is not None else None],
            'incremental_train': [self.incremental_train.to_dict() if self.incremental_train is not None else None]
        }
