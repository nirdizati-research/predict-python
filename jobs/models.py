from django.db import models

from core.constants import REGRESSION, CLASSIFICATION, NEXT_ACTIVITY
from logs.models import Log
from jsonfield import JSONField

CREATED = 'created'
COMPLETED = 'completed'
ERROR = 'error'
RUNNING = 'running'

TYPES = (
    (CLASSIFICATION, 'Classification'),
    (REGRESSION, 'Regression'),
    (NEXT_ACTIVITY, 'Next activity'),
)
STATUSES = (
    (CREATED, 'Created'),
    (COMPLETED, 'Completed'),
    (ERROR, 'Error'),
    (RUNNING, 'Running'),
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
