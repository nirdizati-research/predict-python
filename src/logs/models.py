from django.db import models
from jsonfield.fields import JSONField


class Log(models.Model):
    """A XES log file on disk"""
    name = models.CharField(max_length=200)
    # path = models.CharField(max_length=200) #TODO: SWAP WITH FilePathField
    path = models.FilePathField(path='cache/log_cache/')
    properties = JSONField(default={})

    def to_dict(self):
        return {
            'name': self.name,
            'path': self.path,
            'properties': self.properties
        }
