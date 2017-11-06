from unittest import TestCase

from encoders.boolean_frequency import frequency
from logs.file_service import get_logs


class TestFrequencyGeneral(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]
        self.df = frequency(self.log)

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

        self.assertEqual(1, row['register request'])
        self.assertEqual(0, row['examine casually'])
        self.assertEqual(1, row['check ticket'])
        self.assertEqual(0, row['decide'])
        self.assertEqual(2400.0, row.elapsed_time)
        self.assertEqual(777180.0, row.remaining_time)

    def test_row2(self):
        df = self.df
        row = df[(df.event_nr == 7) & (df.case_id == '5')].iloc[0]

        self.assertEqual(1, row['register request'])
        self.assertEqual(2, row['examine casually'])
        self.assertEqual(2, row['check ticket'])
        self.assertEqual(1, row['decide'])
        self.assertEqual(1, row['reinitiate request'])
        self.assertEqual(0, row['examine thoroughly'])
        self.assertEqual(0, row['pay compensation'])
        self.assertEqual(0, row['reject request'])
        self.assertEqual(687960.0, row.remaining_time)
        self.assertEqual(888480.0, row.elapsed_time)
