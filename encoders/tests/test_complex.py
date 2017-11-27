from unittest import TestCase

from encoders.complex_latest import complex_encode
from logs.file_service import get_logs


class Complex(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]

    def test_shape(self):
        df = complex_encode(self.log, 2)

        self.assertEqual((2, 14), df.shape)
        headers = ["case_id", "event_nr", "remaining_time", "elapsed_time", "prefix_1", "Resource_1", "Costs_1",
                   "Activity_1", "org:resource_1", "prefix_2", "Resource_2", "Costs_2", "Activity_2", "org:resource_2"]
        self.assertListEqual(headers, df.columns.values.tolist())

    def test_prefix1(self):
        df = complex_encode(self.log, 1)

        row1 = df[(df.case_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ["5", 1, 1576440.0, 0.0, "register request", 'Ellen', "50", "register request",
                              "Ellen"])
        row2 = df[(df.case_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ["4", 1, 520920.0, 0.0, "register request", 'Pete', "50", "register request",
                              "Pete"])

    def test_prefix2(self):
        df = complex_encode(self.log, 2)

        row1 = df[(df.case_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ["5", 2, 1485600.0, 90840.0, "register request", 'Ellen', "50", "register request",
                              "Ellen", "examine casually", "Mike", "400", "examine casually", "Mike"])
        row2 = df[(df.case_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ["4", 2, 445080.0, 75840.0, "register request", 'Pete', "50", "register request",
                              "Pete", "check ticket", "Mike", "100", "check ticket", "Mike"])

    def test_prefix5(self):
        df = complex_encode(self.log, 5)

        self.assertEqual(df.shape, (2, 29))

    def test_prefix10(self):
        df = complex_encode(self.log, 10)

        self.assertEqual(df.shape, (1, 54))
