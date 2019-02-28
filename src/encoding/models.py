from django.db import models

from src.encoding.encoding_container import LABEL_ENCODER, ONE_HOT_ENCODER, SIMPLE_INDEX, BOOLEAN, FREQUENCY, COMPLEX, \
    LAST_PAYLOAD

DATA_ENCODING = (
    (LABEL_ENCODER, 'label_encoder'),
    (ONE_HOT_ENCODER, 'one_hot')
)

VALUE_ENCODING = (
    (SIMPLE_INDEX, 'simpleIndex'),
    (BOOLEAN, 'boolean'),
    (FREQUENCY, 'frequency'),
    (COMPLEX, 'complex'),
    (LAST_PAYLOAD, 'lastPayload')
)


class Encoding(models.Model):
    split = models.ForeignKey('pred_models.ModelSplit', on_delete=models.DO_NOTHING, blank=True, null=True)
    data_encoding = models.CharField(choices=DATA_ENCODING, default='label_encoder', max_length=20)
    value_encoding = models.CharField(choices=VALUE_ENCODING, default='label_encoder', max_length=20)
    additional_features = models.BooleanField(default=False)
    temporal_features = models.BooleanField(default=False)
    intercase_features = models.BooleanField(default=False)
    features = models.BinaryField()  # TODO is this correct?
    prefix_len = models.PositiveIntegerField()
    padding = models.BooleanField(default=False)

    def to_dict(self) -> dict:
        return {
            'split': self.split,
            'data_encoding': self.data_encoding,
            'value_encoding': self.value_encoding,
            'additional_features': self.additional_features,
            'temporal_features': self.temporal_features,
            'intercase_features': self.intercase_features,
            'features': self.features,
            'prefix_len': self.prefix_len,
            'padding': self.padding
        }
