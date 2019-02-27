from django.db import models

from src.labelling.label_container import NEXT_ACTIVITY, ATTRIBUTE_STRING, ATTRIBUTE_NUMBER, THRESHOLD_MEAN, \
    THRESHOLD_CUSTOM, REMAINING_TIME, NO_LABEL, DURATION

LABELLING_TYPES = (
    (NEXT_ACTIVITY, 'next_activity'),
    (ATTRIBUTE_STRING, 'attribute_string'),
    (ATTRIBUTE_NUMBER, 'attribute_number'),
    (THRESHOLD_MEAN, 'threshold_mean'),
    (THRESHOLD_CUSTOM, 'threshold_custom'),
    (REMAINING_TIME, 'remaining_time'),
    (DURATION, 'duration'),
    (NO_LABEL, 'no_label')
)

THRESHOLD_TYPES = (
    (THRESHOLD_MEAN, 'threshold_mean'),
    (THRESHOLD_CUSTOM, 'threshold_custom')
)


class Labelling(models.Model):
    type = models.CharField(choices=LABELLING_TYPES, default='attribute_string', max_length=20)
    attribute_name = models.CharField(default='label', max_length=20)
    threshold_type = models.CharField(choices=THRESHOLD_TYPES, default='threshold_mean', max_length=20)
    threshold = models.IntegerField()
    split = models.ForeignKey('split.Split', on_delete=models.DO_NOTHING, blank=True, null=True)

    def to_dict(self):
        return {
            'type': self.type,
            'attribute_name': self.attribute_name,
            'threshold_type': self.threshold_type,
            'threshold': self.threshold,
            'split': self.split
        }
