from unittest import TestCase

from core.constants import BOOLEAN, CLASSIFICATION
from encoders.boolean_frequency import boolean
from encoders.common import LabelContainer, NO_LABEL, encode_label_logs
from encoders.log_util import unique_events
from logs.file_service import get_logs


class TestBooleanGeneral(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]
        self.event_names = unique_events(self.log)
        self.df = boolean(self.log, self.event_names, LabelContainer(add_elapsed_time=True))

    def test_shape(self):
        df = self.df
        names = ['register request', 'examine casually', 'check ticket', 'decide',
                 'reinitiate request', 'examine thoroughly', 'pay compensation',
                 'reject request', 'trace_id', 'event_nr', 'label',
                 'elapsed_time']
        for name in names:
            self.assertIn(name, df.columns.values.tolist())
        self.assertEqual((42, 12), df.shape)

    def test_row(self):
        df = self.df

        row = df[(df.event_nr == 2) & (df.trace_id == '2')].iloc[0]

        self.assertTrue(row['register request'])
        self.assertFalse(row['examine casually'])
        self.assertTrue(row['check ticket'])
        self.assertFalse(row['decide'])
        self.assertEqual(2400.0, row.elapsed_time)
        self.assertEqual(777180.0, row.label)

    def test_row2(self):
        df = self.df
        row = df[(df.event_nr == 5) & (df.trace_id == '2')].iloc[0]

        self.assertTrue(row['register request'])
        self.assertTrue(row['examine casually'])
        self.assertTrue(row['check ticket'])
        self.assertTrue(row['decide'])
        self.assertFalse(row['reinitiate request'])
        self.assertFalse(row['examine thoroughly'])
        self.assertTrue(row['pay compensation'])
        self.assertFalse(row['reject request'])
        self.assertEqual(779580.0, row.elapsed_time)
        self.assertEqual(0.0, row.label)

    def test_no_label(self):
        log = get_logs("log_cache/general_example.xes")[0]
        event_names = unique_events(log)
        df = boolean(log, event_names, LabelContainer(NO_LABEL))
        self.assertEqual((42, 10), df.shape)
        self.assertNotIn('label', df.columns.values.tolist())

    def test_no_elapsed_time(self):
        log = get_logs("log_cache/general_example.xes")[0]
        event_names = unique_events(log)
        df = boolean(log, event_names, LabelContainer())
        self.assertEqual((42, 11), df.shape)
        self.assertNotIn('elapsed_time', df.columns.values.tolist())


class TestBooleanSplit(TestCase):
    def setUp(self):
        test_log = get_logs("log_cache/general_example_test.xes")[0]
        training_log = get_logs("log_cache/general_example_training.xes")[0]
        self.training_df, self.test_df = encode_label_logs(training_log, test_log, BOOLEAN, CLASSIFICATION,
                                                           LabelContainer(add_elapsed_time=True))

    def test_shape(self):
        self.assert_shape(self.training_df, (24, 12))
        self.assert_shape(self.test_df, (18, 12))

    def assert_shape(self, df, shape: tuple):
        names = ['register request', 'examine casually', 'check ticket', 'decide',
                 'reinitiate request', 'examine thoroughly', 'pay compensation',
                 'reject request', 'trace_id', 'event_nr', 'label',
                 'elapsed_time']
        for name in names:
            self.assertIn(name, df.columns.values.tolist())
        self.assertEqual(shape, df.shape)

    def test_row(self):
        df = self.training_df

        row = df[(df.event_nr == 2) & (df.trace_id == '2')].iloc[0]

        self.assertTrue(row['register request'])
        self.assertFalse(row['examine casually'])
        self.assertTrue(row['check ticket'])
        self.assertFalse(row['decide'])
        self.assertEqual(2400.0, row.elapsed_time)
        self.assertEqual(False, row.label)

    def test_row2(self):
        df = self.test_df
        row = df[(df.event_nr == 5) & (df.trace_id == '5')].iloc[0]
        self.assertTrue(row['register request'])
        self.assertTrue(row['examine casually'])
        self.assertTrue(row['check ticket'])
        self.assertTrue(row['decide'])
        self.assertTrue(row['reinitiate request'])
        self.assertFalse(row['examine thoroughly'])
        self.assertFalse(row['pay compensation'])
        self.assertFalse(row['reject request'])
        self.assertEqual(458160.0, row.elapsed_time)
        self.assertEqual(False, row.label)
