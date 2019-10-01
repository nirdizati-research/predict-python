from enum import Enum

from django.db import models
from jsonfield.fields import JSONField

from src.common.models import CommonModel


class LabelTypes(Enum):
    NEXT_ACTIVITY = 'next_activity'
    ATTRIBUTE_STRING = 'attribute_string'
    ATTRIBUTE_NUMBER = 'attribute_number'
    REMAINING_TIME = 'remaining_time'
    DURATION = 'duration'
    NO_LABEL = 'no_label'


class ThresholdTypes(Enum):
    THRESHOLD_MEAN = 'threshold_mean'
    THRESHOLD_CUSTOM = 'threshold_custom'
    NONE = 'none'


#
# CLASSIFICATION_LABELS = [NEXT_ACTIVITY, ATTRIBUTE_STRING, ATTRIBUTE_NUMBER, THRESHOLD_MEAN, THRESHOLD_CUSTOM]
#
# REGRESSION_LABELS = [REMAINING_TIME, ATTRIBUTE_NUMBER]
#
# TIME_SERIES_PREDICTION_LABELS = [NEXT_ACTIVITY]  # TODO: check for using NO_LABEL


LABELLING_TYPE_MAPPINGS = (
    (LabelTypes.NEXT_ACTIVITY.value, 'next_activity'),
    (LabelTypes.ATTRIBUTE_STRING.value, 'attribute_string'),
    (LabelTypes.ATTRIBUTE_NUMBER.value, 'attribute_number'),
    (LabelTypes.REMAINING_TIME.value, 'remaining_time'),
    (LabelTypes.DURATION.value, 'duration'),
    (LabelTypes.NO_LABEL.value, 'no_label')
)

THRESHOLD_TYPE_MAPPINGS = (
    (ThresholdTypes.THRESHOLD_MEAN.value, 'threshold_mean'),
    (ThresholdTypes.THRESHOLD_CUSTOM.value, 'threshold_custom'),
    (ThresholdTypes.NONE.value, 'none')
)


class Labelling(CommonModel):
    type = models.CharField(choices=LABELLING_TYPE_MAPPINGS, default='attribute_string', max_length=max(len(el[1]) for el in LABELLING_TYPE_MAPPINGS)+1, null=True, blank=True)
    attribute_name = models.CharField(default=None, max_length=20, null=True, blank=True)
    threshold_type = models.CharField(choices=THRESHOLD_TYPE_MAPPINGS, default='none', max_length=max(len(el[1]) for el in THRESHOLD_TYPE_MAPPINGS) + 1, null=True, blank=True)
    threshold = models.IntegerField(default=0, null=True, blank=True)
    results = JSONField(default={})

    def to_dict(self):
        return {
            'type': self.type,
            'attribute_name': self.attribute_name,
            'threshold_type': self.threshold_type,
            'threshold': self.threshold,
            'results': self.results
        }
