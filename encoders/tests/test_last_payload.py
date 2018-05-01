from unittest import TestCase

from encoders.complex_last_payload import last_payload
from encoders.label_container import LabelContainer, NO_LABEL
from log_util.event_attributes import unique_events
from logs.file_service import get_logs


class LastPayload(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example_test.xes")[0]
        self.event_names = unique_events(self.log)
        self.label = LabelContainer(add_elapsed_time=True)

    def test_shape(self):
        df = last_payload(self.log, self.event_names, self.label, prefix_length=2)

        self.assertEqual((2, 9), df.shape)
        headers = ['trace_id', 'prefix_1', 'prefix_2', 'Activity_2', 'Costs_2',
                   'Resource_2', 'org:resource_2', 'elapsed_time', 'label']
        self.assertListEqual(headers, df.columns.values.tolist())

    def test_prefix1(self):
        df = last_payload(self.log, self.event_names, self.label, prefix_length=1)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ["5", 'register request', "register request", "50", 'Ellen', "Ellen", 0.0, 1576440.0])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ["4", 'register request', "register request", "50", 'Pete', "Pete", 0.0, 520920.0])

    def test_prefix1_no_label(self):
        df = last_payload(self.log, self.event_names, LabelContainer(NO_LABEL), prefix_length=1)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ["5", 'register request', "register request", "50", 'Ellen', "Ellen"])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ["4", 'register request', "register request", "50", 'Pete', "Pete"])

    def test_prefix1_no_elapsed_time(self):
        df = last_payload(self.log, self.event_names, LabelContainer(add_elapsed_time=False), prefix_length=1)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ["5", 'register request', "register request", "50", 'Ellen', "Ellen", 1576440.0])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ["4", 'register request', "register request", "50", 'Pete', "Pete", 520920.0])

    def test_prefix2(self):
        df = last_payload(self.log, self.event_names, self.label, prefix_length=2)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ["5", 'register request', 'examine casually', "examine casually", "400", "Mike", "Mike",
                              90840.0, 1485600.0])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ["4", 'register request', "check ticket", "check ticket", "100", "Mike", "Mike", 75840.0,
                              445080.0])

    def test_prefix5(self):
        df = last_payload(self.log, self.event_names, self.label, prefix_length=5)

        self.assertEqual(df.shape, (2, 12))

    def test_prefix10(self):
        df = last_payload(self.log, self.event_names, self.label, prefix_length=10)

        self.assertEqual(df.shape, (1, 17))

    def test_prefix10_zero_padding(self):
        df = last_payload(self.log, self.event_names, self.label, prefix_length=10, zero_padding=True)

        self.assertEqual(df.shape, (2, 17))
