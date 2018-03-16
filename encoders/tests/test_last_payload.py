from unittest import TestCase

from encoders.complex_last_payload import last_payload
from encoders.log_util import unique_events
from logs.file_service import get_logs


class LastPayload(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example_test.xes")[0]
        self.event_names, _ = unique_events(self.log)

    def test_shape(self):
        df = last_payload(self.log, self.event_names, prefix_length=2)

        self.assertEqual((2, 9), df.shape)
        headers = ['trace_id', 'remaining_time', 'elapsed_time', 'prefix_1', 'prefix_2', 'Activity_2', 'Costs_2',
                   'Resource_2', 'org:resource_2']
        self.assertListEqual(headers, df.columns.values.tolist())

    def test_prefix1(self):
        df = last_payload(self.log, self.event_names, prefix_length=1)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ["5", 1576440.0, 0.0, 1, "register request", "50", 'Ellen', "Ellen"])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ["4", 520920.0, 0.0, 1, "register request", "50", 'Pete', "Pete"])

    def test_prefix2(self):
        df = last_payload(self.log, self.event_names, prefix_length=2)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ["5", 1485600.0, 90840.0, 1, 2, "examine casually", "400", "Mike", "Mike"])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ["4", 445080.0, 75840.0, 1, 3, "check ticket", "100", "Mike", "Mike"])

    def test_prefix5(self):
        df = last_payload(self.log, self.event_names, prefix_length=5)

        self.assertEqual(df.shape, (2, 12))

    def test_prefix10(self):
        df = last_payload(self.log, self.event_names, prefix_length=10)

        self.assertEqual(df.shape, (1, 17))
