from django.test import TestCase

from core.core import prepare_logs
from logs.models import Log


class LogPrepare(TestCase):
    def test_split_single(self):
        test_log = prepare_logs(split_single())
        self.assertEqual(6, len(test_log))

    def test_split_double(self):
        test_log = prepare_logs(split_double())
        self.assertEqual(2, len(test_log))


def split_single():
    Log.objects.create(name='general-example', path='log_cache/general_example.xes')
    split = dict()
    split['type'] = 'single'
    split['original_log_path'] = 'log_cache/general_example.xes'
    return split


def split_double():
    Log.objects.create(name='general-example-test', path='log_cache/general_example_test.xes')
    Log.objects.create(name='general-example-training', path='log_cache/general_example_training.xes')    
    split = dict()
    split['type'] = 'double'
    split['test_log_path'] = 'log_cache/general_example_test.xes'
    split['training_log_path'] = 'log_cache/general_example_training.xes'
    return split
