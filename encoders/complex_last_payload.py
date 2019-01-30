import pandas as pd
from opyenxes.model import XTrace

from encoders.encoding_container import EncodingContainer
from encoders.label_container import LabelContainer
from encoders.simple_index import add_label_columns, add_labels, get_intercase_attributes

ATTRIBUTE_CLASSIFIER = None


def complex(log: list, label: LabelContainer, encoding: EncodingContainer, additional_columns: list):
    return encode_complex_latest(log, label, encoding, additional_columns, columns_complex, data_complex)


def last_payload(log, label: LabelContainer, encoding: EncodingContainer, additional_columns: list):
    return encode_complex_latest(log, label, encoding, additional_columns, columns_last_payload, data_last_payload)


def encode_complex_latest(log: list, label: LabelContainer, encoding: EncodingContainer, additional_columns: list,
                          column_fun, data_fun):
    columns = column_fun(encoding.prefix_length, additional_columns, label)
    encoded_data = []

    kwargs = get_intercase_attributes(log, label)
    for trace in log:
        if len(trace) <= encoding.prefix_length - 1 and not encoding.is_zero_padding():
            # trace too short and no zero padding
            continue
        if encoding.is_all_in_one():
            for i in range(1, min(encoding.prefix_length + 1, len(trace) + 1)):
                encoded_data.append(
                    trace_to_row(trace, encoding, i, data_fun, columns, additional_columns=additional_columns,
                                 atr_classifier=label.attribute_name, **kwargs))
        else:
            encoded_data.append(
                trace_to_row(trace, encoding, encoding.prefix_length, data_fun, columns, additional_columns=additional_columns,
                             atr_classifier=label.attribute_name, **kwargs))
    return pd.DataFrame(columns=columns, data=encoded_data)


def columns_complex(prefix_length: int, additional_columns: list, label: LabelContainer):
    columns = ['trace_id']
    columns += additional_columns['trace_attributes']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
        for additional_column in additional_columns['event_attributes']:
            columns.append(additional_column + "_" + str(i))
    return add_label_columns(columns, label)


def columns_last_payload(prefix_length: int, additional_columns: list, label: LabelContainer):
    columns = ['trace_id']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
    for additional_column in additional_columns['event_attributes']:
        columns.append(additional_column + "_" + str(i))
    return add_label_columns(columns, label)


def data_complex(trace: list, prefix_length: int, additional_columns: list):
    """Creates list in form [1, value1, value2, 2, ...]

    Appends values in additional_columns
    """
    data = [ trace.attributes.get(att, '0') for att in additional_columns['trace_attributes'] ]
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = event["concept:name"]
        data.append(event_name)

        for att in additional_columns['event_attributes']:
            data.append(event[att])

    return data


def data_last_payload(trace: list, prefix_length: int, additional_columns: list):
    """Creates list in form [1, 2, value1, value2,]

    Event name index of the position they are in event_names
    Appends values in additional_columns
    """
    data = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = event['concept:name']
        data.append(event_name)

    # Attributes of last event
    #TODO: this is very strange
    for att in additional_columns['event_attributes']:
        if prefix_length - 1 >= len(trace):
            value = '0'
        else:
            value = trace[prefix_length - 1][att]
        data.append(value)
    return data


def trace_to_row(trace: XTrace, encoding: EncodingContainer, event_index: int, data_fun, columns, atr_classifier=None,
                 label=None,
                 executed_events=None, resources_used=None, new_traces=None, additional_columns=None):
    trace_row = [trace.attributes["concept:name"]]
    # prefix_length - 1 == index
    trace_row += data_fun(trace, event_index, additional_columns)
    if encoding.is_zero_padding() or encoding.is_all_in_one():
        trace_row += ['0' for _ in range(len(trace_row) + 1, len(columns) - 1)]
    trace_row += add_labels(label, event_index, trace, atr_classifier=atr_classifier,
                            executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    return trace_row
