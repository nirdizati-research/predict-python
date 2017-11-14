from django.db import models

from logs.models import Log
from jsonfield import JSONField


class BaseModel(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Job(BaseModel):
    config = JSONField()
    status = models.CharField(default='created', max_length=200)
    result = JSONField(default={})
    type = models.CharField(max_length=20)
