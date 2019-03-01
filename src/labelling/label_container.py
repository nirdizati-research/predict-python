from collections import namedtuple

from src.labelling.models import LabelTypes, ThresholdTypes


class LabelContainer(namedtuple('LabelContainer', ['type', 'attribute_name', 'threshold_type', 'threshold',
                                                   'add_remaining_time', 'add_elapsed_time', 'add_executed_events',
                                                   'add_resources_used', 'add_new_traces'])):
    """Inner object describing labelling state.
    For no labelling use NO_LABEL

    This is a horrible hack and should be split into a label container and a container for encoding options, like
    what to add to the encoded log.
    """

    def __new__(cls, type: str = LabelTypes.REMAINING_TIME.value, attribute_name: str = None,
                threshold_type: str = ThresholdTypes.THRESHOLD_MEAN.value,
                threshold: int = 0, add_remaining_time: bool = False, add_elapsed_time: bool = False,
                add_executed_events: bool = False, add_resources_used: bool = False, add_new_traces: bool = False):
        # noinspection PyArgumentList
        return super(LabelContainer, cls).__new__(cls, type, attribute_name, threshold_type, threshold,
                                                  add_remaining_time, add_elapsed_time, add_executed_events,
                                                  add_resources_used, add_new_traces)
