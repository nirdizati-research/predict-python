"""Add a label to an encoded log

Steps
1. Encoded log
2. Define labelling types
    remaining_time
    next_activity
    attribute_number
    attribute_string

Encoding
Post processing if needed


def labelling
    enc = encodedLog
    if reg
        rename column
    elif class
        rename and map column to bool if
"""
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
    def __new__(cls, type=REMAINING_TIME, attribute_name=None, threshold_type=THRESHOLD_MEAN, threshold=0,
                add_remaining_time=False, add_elapsed_time=False):
        return super(LabelContainer, cls).__new__(cls, type, attribute_name, threshold_type, threshold,
                                                  add_remaining_time, add_elapsed_time)

