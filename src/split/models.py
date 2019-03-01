from enum import Enum

from django.db import models

from src.logs.models import Log


class SplitTypes(Enum):
    SPLIT_SINGLE = 'single'
    SPLIT_DOUBLE = 'double'


SPLIT_TYPE_MAPPINGS = (
    (SplitTypes.SPLIT_SINGLE, 'single'),
    (SplitTypes.SPLIT_DOUBLE, 'double'),
)


class Split(models.Model):
    """Container of Split to be shown in frontend"""
    type = models.CharField(choices=SPLIT_TYPE_MAPPINGS, default='single', max_length=20)
    original_log = models.ForeignKey(Log, on_delete=models.DO_NOTHING, related_name='original_log', blank=True,
                                     null=True)
    test_log = models.ForeignKey(Log, on_delete=models.CASCADE, related_name='test_log', blank=True, null=True)
    training_log = models.ForeignKey(Log, on_delete=models.CASCADE, related_name='training_log', blank=True, null=True)

    def to_dict(self) -> dict:
        split = {
            'id': self.id,
            'type': self.type
        }
        if self.type == 'single':
            split['original_log_path'] = self.original_log.path
        else:
            split['test_log_path'] = self.test_log.path
            split['training_log_path'] = self.training_log.path
        return split
