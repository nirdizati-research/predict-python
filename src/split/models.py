from django.db import models
from jsonfield.fields import JSONField

SPLIT_TYPES = (
    ('single', 'Single'),
    ('double', 'Double'),
)


class Split(models.Model):
    """Container of Split to be shown in frontend"""
    config = JSONField(default={})
    type = models.CharField(choices=SPLIT_TYPES, default='single', max_length=20)
    original_log = models.ForeignKey('logs.Log', on_delete=models.DO_NOTHING, related_name='original_log', blank=True,
                                     null=True)
    test_log = models.ForeignKey('logs.Log', on_delete=models.CASCADE, related_name='test_log', blank=True, null=True)
    training_log = models.ForeignKey('logs.Log', on_delete=models.CASCADE, related_name='training_log', blank=True,
                                     null=True)

    def to_dict(self) -> dict:
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
