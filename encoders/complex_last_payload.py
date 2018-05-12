import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier
from opyenxes.model import XTrace

from encoders.encoding_container import EncodingContainer
from encoders.label_container import LabelContainer
from encoders.simple_index import add_label_columns, add_labels, setup_attribute_classifier, get_intercase_attributes

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])
ATTRIBUTE_CLASSIFIER = None


def complex(log: list, label: LabelContainer, encoding: EncodingContainer, additional_columns: list):
    return encode_complex_latest(log, label, encoding, additional_columns, columns_complex, data_complex)


def last_payload(log, label: LabelContainer, encoding: EncodingContainer, additional_columns: list):
    return encode_complex_latest(log, label, encoding, additional_columns, columns_last_payload, data_last_payload)


def encode_complex_latest(log: list, label: LabelContainer, encoding: EncodingContainer, additional_columns: list,
                          column_fun, data_fun):
    columns = column_fun(encoding.prefix_length, additional_columns, label)
    encoded_data = []

    atr_classifier = setup_attribute_classifier(label)
    kwargs = get_intercase_attributes(log, label)
    for trace in log:
        if len(trace) <= encoding.prefix_length - 1 and not encoding.is_zero_padding():
            # trace too short and no zero padding
            continue
        if encoding.is_all_in_one():
            for i in range(1, encoding.prefix_length + 1):
                encoded_data.append(
                    trace_to_row(trace, encoding, i, data_fun, additional_columns=additional_columns,
                                 atr_classifier=atr_classifier, **kwargs))
        else:
            encoded_data.append(
                trace_to_row(trace, encoding, encoding.prefix_length, data_fun, additional_columns=additional_columns,
                             atr_classifier=atr_classifier, **kwargs))

    return pd.DataFrame(columns=columns, data=encoded_data)


def columns_complex(prefix_length: int, additional_columns: list, label: LabelContainer):
    columns = ['trace_id']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
        for additional_column in additional_columns:
            columns.append(additional_column + "_" + str(i))
    return add_label_columns(columns, label)


def columns_last_payload(prefix_length: int, additional_columns: list, label: LabelContainer):
    columns = ['trace_id']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
    for additional_column in additional_columns:
        columns.append(additional_column + "_" + str(i))
    return add_label_columns(columns, label)


def data_complex(trace: list, prefix_length: int, additional_columns: list):
    """Creates list in form [1, value1, value2, 2, ...]

    Appends values in additional_columns
    """
    data = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = CLASSIFIER.get_class_identity(event)
        data.append(event_name)

        for att in additional_columns:
            # Basically XEventAttributeClassifier
            value = event.get_attributes().get(att).get_value()
            data.append(value)

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
        event_name = CLASSIFIER.get_class_identity(event)
        data.append(event_name)

    # Attributes of last event
    for att in additional_columns:
        # Basically XEventAttributeClassifier
        if prefix_length - 1 >= len(trace):
            value = '0'
        else:
            event_attrs = trace[prefix_length - 1].get_attributes()
            value = event_attrs.get(att).get_value()
        data.append(value)
    return data


def trace_to_row(trace: XTrace, encoding: EncodingContainer, event_index: int, data_fun, atr_classifier=None,
                 label=None,
                 executed_events=None, resources_used=None, new_traces=None, additional_columns=None):
    zero_count = get_zero_count(encoding, event_index, len(trace), len(additional_columns))
    trace_row = []
    trace_name = CLASSIFIER.get_class_identity(trace)
    trace_row.append(trace_name)
    # prefix_length - 1 == index
    trace_row += data_fun(trace, event_index, additional_columns)
    if encoding.is_zero_padding() or encoding.is_all_in_one():
        trace_row += ['0' for _ in range(0, zero_count)]
    trace_row += add_labels(label, event_index, trace, atr_classifier=atr_classifier,
                            executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    return trace_row


def get_zero_count(encoding: EncodingContainer, event_index: int, trace_len: int, add_columns_len: int):
    # Don't know what to call these
    a = encoding.prefix_length - event_index
    b = encoding.prefix_length - trace_len
    if encoding.is_all_in_one() and encoding.is_complex():
        if a < b:
            a = b
        zero_count = a * (1 + add_columns_len)
    elif encoding.is_all_in_one():
        zero_count = a if a > b else b
        if zero_count > 0:
            zero_count + add_columns_len
    elif encoding.is_zero_padding() and encoding.is_complex():
        zero_count = b * (1 + add_columns_len)
    elif encoding.is_zero_padding():
        zero_count = b
        if zero_count > 0:
            zero_count + add_columns_len
    else:
        zero_count = 0
    return zero_count
