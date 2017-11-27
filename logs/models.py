from django.db import models
from jsonfield.fields import JSONField

from logs.file_service import get_logs

TYPES = (
    ('single', 'Single'),
    ('double', 'Double'),
)


class Log(models.Model):
    """A XES log file on disk"""
    name = models.CharField(max_length=200)
    path = models.CharField(max_length=200)

    def get_file(self):
        """Read and parse log from filesystem"""
        return get_logs(self.path)


class Split(models.Model):
    """Container of Log to be shown in frontend"""
    original_log = models.ForeignKey('Log', on_delete=models.DO_NOTHING)
    config = JSONField(default={})
    type = models.CharField(choices=TYPES, default='single', max_length=20)
