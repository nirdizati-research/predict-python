from django.db import models
from jsonfield import JSONField

from core.constants import REGRESSION, CLASSIFICATION, LABELLING, UPDATE
from encoders.encoding_container import EncodingContainer
from encoders.label_container import LabelContainer
from logs.models import Split

CREATED = 'created'
COMPLETED = 'completed'
ERROR = 'error'
RUNNING = 'running'

TYPES = (
    (CLASSIFICATION, 'Classification'),
    (REGRESSION, 'Regression'),
    (LABELLING, 'Labelling'),
    (UPDATE, 'Update')
)
STATUSES = (
    (CREATED, 'Created'),
    (COMPLETED, 'Completed'),
    (ERROR, 'Error'),
    (RUNNING, 'Running')
)


class BaseModel(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Job(BaseModel):
    config = JSONField()
    error = models.CharField(default='', max_length=200)
    status = models.CharField(choices=STATUSES, default=CREATED, max_length=20)
    result = JSONField(default={})
    type = models.CharField(choices=TYPES, max_length=20)
    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)

    def to_dict(self) -> dict:
        job = dict(self.config)
        job['type'] = self.type
        job['split'] = self.split.to_dict()
        job['label'] = LabelContainer(**self.config['label'])
        job['encoding'] = EncodingContainer(**self.config['encoding'])
        return job
