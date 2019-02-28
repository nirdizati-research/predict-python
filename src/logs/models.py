from django.db import models
from jsonfield.fields import JSONField

from src.utils.file_service import get_log


class Log(models.Model):
    """A XES log file on disk"""
    name = models.CharField(max_length=200)
    # path = models.CharField(max_length=200) #TODO: SWAP WITH FilePathField
    path = models.FilePathField(path='cache/log_cache/')
    properties = JSONField(default={})

    def get_file(self):
        """Read and parse log from filesystem"""
        return get_log(self.path)
