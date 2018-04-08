from django.test import TestCase
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from logs.splitting import prepare_logs

classifier = XEventAttributeClassifier("Trace", ["concept:name"])


class Split(TestCase):
    def test_split_single(self):
        training_log, test_log = prepare_logs(split_single())
        self.assertEqual(4, len(training_log))
        self.assertEqual(2, len(test_log))

    def test_split_double(self):
        training_log, test_log = prepare_logs(split_double())
        self.assertEqual(4, len(training_log))
        self.assertEqual(2, len(test_log))


class SplitSingle(TestCase):
    def test_size(self):
        split = split_single()
        split['config'] = {'test_size': 0.5}
        training_log, test_log = prepare_logs(split)
        self.assertEqual(3, len(training_log))
        self.assertEqual(3, len(test_log))

    def test_sequential(self):
        split = split_single()
        split['config'] = {'split_type': 'split_sequential'}
        training_log, test_log = prepare_logs(split)
        training_names = trace_names(training_log)
        test_names = trace_names(test_log)

        self.assertListEqual(['3', '2', '1', '6'], training_names)
        self.assertListEqual(['5', '4'], test_names)

    def test_random(self):
        split = split_single()
        split['config'] = {'split_type': 'split_random'}
        training_log1, _ = prepare_logs(split)
        training_log2, _ = prepare_logs(split)
        training_names1 = trace_names(training_log1)
        training_names2 = trace_names(training_log2)

        self.assertNotEqual(training_names1, training_names2)

    def test_temporal(self):
        split = split_single()
        split['config'] = {'split_type': 'split_temporal'}
        training_log, test_log = prepare_logs(split)

        training_names = trace_names(training_log)
        test_names = trace_names(test_log)

        self.assertListEqual(sorted(['1', '2', '3', '5']), sorted(training_names))
        self.assertListEqual(sorted(['6', '4']), sorted(test_names))


def trace_names(log):
    """Get trace names"""
    names = []
    for trace in log:
        name = classifier.get_class_identity(trace)
        names.append(name)
    return names


def split_single():
    split = dict()
    split['config'] = dict()
    split['type'] = 'single'
    split['original_log_path'] = 'log_cache/general_example.xes'
    return split


def split_double():
    split = dict()
    split['config'] = dict()
    split['type'] = 'double'
    split['test_log_path'] = 'log_cache/general_example_test.xes'
    split['training_log_path'] = 'log_cache/general_example_training.xes'
    return split
