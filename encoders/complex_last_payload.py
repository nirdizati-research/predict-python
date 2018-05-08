import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.label_container import LabelContainer, ATTRIBUTE_STRING, ATTRIBUTE_NUMBER
from encoders.simple_index import add_label_columns, add_labels
from log_util.log_metrics import events_by_date, resources_by_date, new_trace_start

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])
ATTRIBUTE_CLASSIFIER = None


def complex(log, label: LabelContainer, additional_columns: list, prefix_length=1, zero_padding=False, ):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    return encode_complex_latest(log, label, prefix_length, additional_columns, columns_complex, data_complex,
                                 zero_padding, is_complex=True)


def last_payload(log, label: LabelContainer, additional_columns: list, prefix_length=1, zero_padding=False):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    return encode_complex_latest(log, label, prefix_length, additional_columns, columns_last_payload, data_last_payload,
                                 zero_padding)


def encode_complex_latest(log, label: LabelContainer, prefix_length: int, additional_columns: list, column_fun,
                          data_fun, zero_padding: bool, is_complex=False):
    columns = column_fun(prefix_length, additional_columns, label)
    encoded_data = []

    # Create classifier only once
    if label.type == ATTRIBUTE_STRING or label.type == ATTRIBUTE_NUMBER:
        global ATTRIBUTE_CLASSIFIER
        ATTRIBUTE_CLASSIFIER = XEventAttributeClassifier("Attr class", [label.attribute_name])
    # Expensive operations
    executed_events = events_by_date([log]) if label.add_executed_events else None
    resources_used = resources_by_date([log]) if label.add_resources_used else None
    new_traces = new_trace_start([log]) if label.add_new_traces else None
    for trace in log:
        if zero_padding and is_complex:
            zero_count = (prefix_length - len(trace)) * (1 + len(additional_columns))
        elif zero_padding:
            zero_count = prefix_length - len(trace)
            if zero_count > 0:
                zero_count + len(additional_columns)
        elif len(trace) <= prefix_length - 1:
            # no padding, skip this trace
            continue
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)
        # prefix_length - 1 == index
        trace_row += data_fun(trace, prefix_length, additional_columns)
        if zero_padding:
            trace_row += ['0' for _ in range(0, zero_count)]
        trace_row += add_labels(label, prefix_length, trace, ATTRIBUTE_CLASSIFIER=ATTRIBUTE_CLASSIFIER,
                                executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
        encoded_data.append(trace_row)

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
