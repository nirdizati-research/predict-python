from django.db import models
from jsonfield.fields import JSONField

from src.common.models import CommonModel


class Log(CommonModel):
    """A XES log file on disk"""
    name = models.CharField(max_length=200)
    path = models.FilePathField()
    properties = JSONField(default={})

    def to_dict(self):
        return {
            'name': self.name,
            'path': self.path,
            'properties': self.properties
        }
