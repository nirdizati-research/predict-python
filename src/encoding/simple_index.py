import pandas as pd
from pandas import DataFrame
from pm4py.objects.log.log import Trace, EventLog

from src.encoding.encoder import PREFIX_
from src.encoding.models import Encoding, TaskGenerationTypes
from src.labelling.common import compute_label_columns, get_intercase_attributes, add_labels
from src.labelling.models import Labelling

ATTRIBUTE_CLASSIFIER = None


def simple_index(log: EventLog, labelling: Labelling, encoding: Encoding) -> DataFrame:
    columns = _compute_columns(encoding.prefix_length)
    normal_columns_number = len(columns)
    columns = compute_label_columns(columns, encoding, labelling)
    encoded_data = []
    kwargs = get_intercase_attributes(log, encoding)
    for trace in log:
        if len(trace) <= encoding.prefix_length - 1 and not encoding.padding:
            # trace too short and no zero padding
            continue
        if encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
            for event_index in range(1, min(encoding.prefix_length + 1, len(trace) + 1)):
                encoded_data.append(add_trace_row(trace, encoding, labelling, event_index, normal_columns_number,
                                                  labelling.attribute_name, **kwargs))
        else:
            encoded_data.append(add_trace_row(trace, encoding, labelling, encoding.prefix_length, normal_columns_number,
                                              labelling.attribute_name, **kwargs))

    return pd.DataFrame(columns=columns, data=encoded_data)


def add_trace_row(trace: Trace, encoding: Encoding, labelling: Labelling, event_index: int, column_len: int,
                  attribute_classifier=None, executed_events=None, resources_used=None, new_traces=None):
    """Row in data frame"""
    trace_row = [trace.attributes['concept:name']]
    trace_row += _trace_prefixes(trace, event_index)
    if encoding.padding or encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
        trace_row += [0 for _ in range(len(trace_row), column_len)]
    trace_row += add_labels(encoding, labelling, event_index, trace, attribute_classifier=attribute_classifier,
                            executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    return trace_row


def _trace_prefixes(trace: Trace, prefix_length: int) -> list:
    """List of indexes of the position they are in event_names

    """
    prefixes = []
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = event['concept:name']
        prefixes.append(event_name)
    return prefixes


def _compute_columns(prefix_length: int) -> list:
    """trace_id, prefixes, any other columns, label

    """
    return ["trace_id"] + [PREFIX_ + str(i + 1) for i in range(0, prefix_length)]
