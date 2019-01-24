from django.db import models
from jsonfield.fields import JSONField

from logs.file_service import get_logs_old

TYPES = (
    ('single', 'Single'),
    ('double', 'Double'),
)


class Log(models.Model):
    """A XES log file on disk"""
    name = models.CharField(max_length=200)
    path = models.CharField(max_length=200)
    properties = JSONField(default={})

    def get_file(self):
        """Read and parse log from filesystem"""
        return get_logs_old(self.path)


class Split(models.Model):
    """Container of Log to be shown in frontend"""
    config = JSONField(default={})
    type = models.CharField(choices=TYPES, default='single', max_length=20)
    original_log = models.ForeignKey('Log', on_delete=models.DO_NOTHING, related_name='original_log', blank=True,
                                     null=True)
    test_log = models.ForeignKey('Log', on_delete=models.CASCADE, related_name='test_log', blank=True, null=True)
    training_log = models.ForeignKey('Log', on_delete=models.CASCADE, related_name='training_log', blank=True,
                                     null=True)

    def to_dict(self):
        split = dict()
        split['id'] = self.id
        split['type'] = self.type
        split['config'] = self.config
        if self.type == 'single':
            split['original_log_path'] = self.original_log.path
        else:
            split['test_log_path'] = self.test_log.path
            split['training_log_path'] = self.training_log.path
        return split
