from collections import namedtuple

NEXT_ACTIVITY = 'next_activity'
REMAINING_TIME = 'remaining_time'
ATTRIBUTE_NUMBER = 'attribute_number'
ATTRIBUTE_STRING = 'attribute_string'
NO_LABEL = 'no_label'

THRESHOLD_MEAN = 'threshold_mean'
THRESHOLD_CUSTOM = 'threshold_custom'


class LabelContainer(namedtuple('LabelContainer', ["type", "attribute_name", "threshold_type", "threshold",
                                                   "add_remaining_time", "add_elapsed_time"])):
    """Inner object describing labelling state.
    For no labelling use NO_LABEL
    """

    def __new__(cls, type=REMAINING_TIME, attribute_name=None, threshold_type=THRESHOLD_MEAN, threshold=0,
                add_remaining_time=False, add_elapsed_time=False):
        return super(LabelContainer, cls).__new__(cls, type, attribute_name, threshold_type, threshold,
                                                  add_remaining_time, add_elapsed_time)
