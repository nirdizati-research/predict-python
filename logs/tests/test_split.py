from django.test import TestCase

from logs.splitting import prepare_logs


class Split(TestCase):
    def test_split_single(self):
        training_log, test_log = prepare_logs(split_single())
        self.assertEqual(4, len(training_log))
        self.assertEqual(2, len(test_log))

    def test_split_double(self):
        training_log, test_log = prepare_logs(split_double())
        self.assertEqual(4, len(training_log))
        self.assertEqual(2, len(test_log))


def split_single():
    split = dict()
    split['type'] = 'single'
    split['original_log_path'] = 'log_cache/general_example.xes'
    return split


def split_double():
    split = dict()
    split['type'] = 'double'
    split['test_log_path'] = 'log_cache/general_example_test.xes'
    split['training_log_path'] = 'log_cache/general_example_training.xes'
    return split
