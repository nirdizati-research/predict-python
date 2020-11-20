from django.contrib.postgres.fields import JSONField
from django.db import models

from src.common.models import CommonModel


class Log(CommonModel):
    """A XES log file on disk"""
    name = models.CharField(max_length=200)
    path = models.FilePathField(path='cache/log_cache/')
    properties = JSONField(default=dict)

    def to_dict(self):
        return {
            'name': self.name,
            'path': self.path,
            'properties': self.properties
        }
