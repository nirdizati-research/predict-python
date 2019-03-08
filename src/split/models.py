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


class SplitOrderingMethods(Enum):
    SPLIT_SEQUENTIAL = 'sequential'
    SPLIT_TEMPORAL = 'temporal'
    SPLIT_RANDOM = 'random'
    SPLIT_STRICT_TEMPORAL = 'strict_temporal'


SPLIT_ORDERING_METHOD_MAPPINGS = (
    (SplitOrderingMethods.SPLIT_SEQUENTIAL.value, 'sequential'),
    (SplitOrderingMethods.SPLIT_TEMPORAL.value, 'temporal'),
    (SplitOrderingMethods.SPLIT_RANDOM.value, 'random'),
    (SplitOrderingMethods.SPLIT_STRICT_TEMPORAL.value, 'strict_temporal')
)


class Split(CommonModel):
    """Container of Split to be shown in frontend"""
    type = models.CharField(choices=SPLIT_TYPE_MAPPINGS, default='single', max_length=20)
    original_log = models.ForeignKey(Log, on_delete=models.DO_NOTHING, related_name='original_log', blank=True,
                                     null=True)
    test_size = models.FloatField(default=0.2, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], blank=True,
                                  null=True)
    splitting_method = models.CharField(choices=SPLIT_ORDERING_METHOD_MAPPINGS, default='sequential', max_length=20)
    train_log = models.ForeignKey(Log, on_delete=models.CASCADE, related_name='training_log', blank=True, null=True)
    test_log = models.ForeignKey(Log, on_delete=models.CASCADE, related_name='test_log', blank=True, null=True)
    additional_columns = models.CharField(max_length=30, blank=True, null=True)

    def to_dict(self) -> dict:
        temp = {
            'id': self.pk,
            'type': self.type,
            'test_size': self.test_size,
            'splitting_method': self.splitting_method
        }
        if self.type == 'single':
            temp['original_log_path'] = self.original_log.path
        else:
            temp['test_log_path'] = self.test_log.path
            temp['train_log_path'] = self.train_log.path
        return temp
