from enum import Enum

from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

from src.common.models import CommonModel
from src.logs.models import Log


class SplitTypes(Enum):
    SPLIT_SINGLE = 'single'
    SPLIT_DOUBLE = 'double'


SPLIT_TYPE_MAPPINGS = (
    (SplitTypes.SPLIT_SINGLE, 'single'),
    (SplitTypes.SPLIT_DOUBLE, 'double'),
)


class SplittingMethods(Enum):
    SPLIT_SEQUENTIAL = 'sequential'
    SPLIT_TEMPORAL = 'temporal'
    SPLIT_RANDOM = 'random'
    SPLIT_STRICT_TEMPORAL = 'strict_temporal'


SPLITTING_METHOD_MAPPINGS = (
    (SplittingMethods.SPLIT_SEQUENTIAL.value, 'sequential'),
    (SplittingMethods.SPLIT_TEMPORAL.value, 'temporal'),
    (SplittingMethods.SPLIT_RANDOM.value, 'random'),
    (SplittingMethods.SPLIT_STRICT_TEMPORAL.value, 'strict_temporal')
)


class Split(CommonModel):
    """Container of Split to be shown in frontend"""
    type = models.CharField(choices=SPLIT_TYPE_MAPPINGS, default='single', max_length=20)
    original_log = models.ForeignKey(Log, on_delete=models.DO_NOTHING, related_name='original_log', blank=True,
                                     null=True)
    test_size = models.FloatField(default=0.2, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], blank=True,
                                  null=True)
    splitting_method = models.CharField(choices=SPLITTING_METHOD_MAPPINGS, default='split_sequential', max_length=20)
    test_log = models.ForeignKey(Log, on_delete=models.CASCADE, related_name='test_log', blank=True, null=True)
    training_log = models.ForeignKey(Log, on_delete=models.CASCADE, related_name='training_log', blank=True, null=True)

    def to_dict(self) -> dict:
        split = {
            'type': self.type,
            'test_size': self.test_size,
            'splitting_method': self.splitting_method
        }
        if self.type == 'single':
            split['original_log_path'] = self.original_log.path
        else:
            split['training_log_path'] = self.training_log.path
            split['test_log_path'] = self.test_log.path
        return split
