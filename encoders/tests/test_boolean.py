from unittest import TestCase

from encoders.boolean_frequency import boolean
from logs.file_service import get_logs


class TestBooleanGeneral(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]
        self.df = boolean(self.log)

    def test_shape(self):
        df = self.df
        names = ['register request', 'examine casually', 'check ticket', 'decide',
                 'reinitiate request', 'examine thoroughly', 'pay compensation',
                 'reject request', 'case_id', 'event_nr', 'remaining_time',
                 'elapsed_time']
        for name in names:
            self.assertIn(name, df.columns.values.tolist())
        self.assertEqual((42, 12), df.shape)

    def test_row(self):
        df = self.df

        row = df[(df.event_nr == 2) & (df.case_id == '2')].iloc[0]

        self.assertTrue(row['register request'])
        self.assertFalse(row['examine casually'])
        self.assertTrue(row['check ticket'])
        self.assertFalse(row['decide'])
        self.assertEqual(2400.0, row.elapsed_time)
        self.assertEqual(777180.0, row.remaining_time)

    def test_row2(self):
        df = self.df
        row = df[(df.event_nr == 5) & (df.case_id == '2')].iloc[0]

        self.assertTrue(row['register request'])
        self.assertTrue(row['examine casually'])
        self.assertTrue(row['check ticket'])
        self.assertTrue(row['decide'])
        self.assertFalse(row['reinitiate request'])
        self.assertFalse(row['examine thoroughly'])
        self.assertTrue(row['pay compensation'])
        self.assertFalse(row['reject request'])
        self.assertEqual(779580.0, row.elapsed_time)
        self.assertEqual(0.0, row.remaining_time)
