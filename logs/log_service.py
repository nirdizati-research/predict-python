from collections import defaultdict
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier


def events_by_date(logs):
    """Creates dict of events by timestamp

    Expected result
    {'2010-12-30': 7, '2011-01-06': 8}
    """
    classifier = XEventAttributeClassifier("Resource", ["time:timestamp"])
    stamp_dict = defaultdict(lambda: 0)
    for log in logs:
        for trace in log:
            for event in trace:
                timestamp = classifier.get_class_identity(event)
                date = timestamp.split("T")[0]
                stamp_dict[date] += 1
    return stamp_dict
