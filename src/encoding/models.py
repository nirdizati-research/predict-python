from enum import Enum

from django.db import models
from jsonfield.fields import JSONField


class DataEncodings(Enum):
    LABEL_ENCODER = 'label_encoder'
    ONE_HOT_ENCODER = 'one_hot'


class ValueEncodings(Enum):
    SIMPLE_INDEX = 'simpleIndex'
    BOOLEAN = 'boolean'
    FREQUENCY = 'frequency'
    COMPLEX = 'complex'
    LAST_PAYLOAD = 'lastPayload'


DATA_ENCODING_MAPPINGS = (
    (DataEncodings.LABEL_ENCODER, 'label_encoder'),
    (DataEncodings.ONE_HOT_ENCODER, 'one_hot')
)

VALUE_ENCODING_MAPPINGS = (
    (ValueEncodings.SIMPLE_INDEX, 'simpleIndex'),
    (ValueEncodings.BOOLEAN, 'boolean'),
    (ValueEncodings.FREQUENCY, 'frequency'),
    (ValueEncodings.COMPLEX, 'complex'),
    (ValueEncodings.LAST_PAYLOAD, 'lastPayload')
)


class Encoding(models.Model):
    data_encoding = models.CharField(choices=DATA_ENCODING_MAPPINGS, default='label_encoder', max_length=20)
    value_encoding = models.CharField(choices=VALUE_ENCODING_MAPPINGS, default='label_encoder', max_length=20)
    additional_features = models.BooleanField(default=False)
    temporal_features = models.BooleanField(default=False)
    intercase_features = models.BooleanField(default=False)
    features = JSONField(default={})  # TODO is this correct?
    prefix_len = models.PositiveIntegerField()
    padding = models.BooleanField(default=False)

    def to_dict(self) -> dict:
        return {
            'data_encoding': self.data_encoding,
            'value_encoding': self.value_encoding,
            'additional_features': self.additional_features,
            'temporal_features': self.temporal_features,
            'intercase_features': self.intercase_features,
            'features': self.features,
            'prefix_len': self.prefix_len,
            'padding': self.padding
        }
