from django.db import models
from jsonfield import JSONField

from src.core.constants import CLASSIFICATION, REGRESSION, LABELLING, UPDATE
from src.encoding.encoding_container import EncodingContainer
from src.labelling.label_container import LabelContainer
from src.split.models import Split

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


class JobBase(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Job(JobBase):
    config = JSONField()
    error = models.CharField(default='', max_length=200)
    status = models.CharField(choices=STATUSES, default=CREATED, max_length=20)
    result = JSONField(default={})
    type = models.CharField(choices=TYPES, max_length=20)
    split = models.ForeignKey(Split, on_delete=models.DO_NOTHING, null=True)

    def to_dict(self) -> dict:
        return {
            **self.config,
            'type': self.type,
            'split': self.split.to_dict(),
            'label': LabelContainer(**self.config['label']),
            'encoding': EncodingContainer(**self.config['encoding'])
        }
