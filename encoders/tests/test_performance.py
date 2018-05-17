import time
import unittest
from unittest import TestCase

from core.constants import REGRESSION
from encoders.common import encode_label_log
from encoders.encoding_container import SIMPLE_INDEX, BOOLEAN, FREQUENCY, COMPLEX, LAST_PAYLOAD, EncodingContainer, \
    ZERO_PADDING, ALL_IN_ONE
from encoders.label_container import *
from log_util.event_attributes import unique_events, get_global_event_attributes
from logs.file_service import get_logs


class TestEncoding(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/repairExample.xes")[0]
        # self.log = get_logs("log_cache/BPI Challenge 2017.xes.gz")[0]
        self.label = LabelContainer(NO_LABEL)
        self.add_col = get_global_event_attributes(self.log)

    def method_self(self, encoding):
        start_time = time.time()
        # log = get_logs("log_cache/repairExample.xes")[0]
        event_names = unique_events(self.log)
        encode_label_log(self.log, encoding, REGRESSION, self.label, event_names=event_names,
                         prefix_length=180, zero_padding=True, additional_columns=self.add_col)
        print("Total for %s %s seconds" % (encoding, time.time() - start_time))

    # This is test
    def performance(self):
        encs = [SIMPLE_INDEX, BOOLEAN, FREQUENCY, COMPLEX, LAST_PAYLOAD]

        # self.method_self(COMPLEX)
        for e in encs:
            self.method_self(e)


@unittest.skip("performance test not needed normally")
class TestAgainstNirdizatiTraining(TestCase):
    def do_test(self, encoding):
        start_time = time.time()
        # log = get_logs("log_cache/general_example.xes")[0]
        log = get_logs("log_cache/Sepsis Cases - Event Log.xes")[0]
        label = LabelContainer(REMAINING_TIME, add_elapsed_time=True)
        encoding = EncodingContainer(encoding, prefix_length=185, generation_type=ALL_IN_ONE,
                                     padding=ZERO_PADDING)
        event_names = unique_events(log)
        log = encode_label_log(log, encoding, REGRESSION, label, event_names=event_names)
        print(log.shape)
        print("Total for %s %s seconds" % (encoding, time.time() - start_time))

    def test_performance(self):
        encs = [SIMPLE_INDEX, BOOLEAN, FREQUENCY]

        # self.method_self(COMPLEX)
        for e in encs:
            self.do_test(e)


@unittest.skip("performance test not needed normally")
class TestTraceLengthTime(TestCase):
    def setUp(self):
        self.label = LabelContainer(NO_LABEL)

    def do_test(self, encoding, log_path):
        start_time = time.time()
        log = get_logs(log_path)[0]
        add_col = get_global_event_attributes(log)
        event_names = unique_events(log)
        encoding = EncodingContainer(encoding, prefix_length=20, padding=ZERO_PADDING)
        log = encode_label_log(log, encoding, REGRESSION, self.label, event_names=event_names,
                               additional_columns=add_col)
        print(log.shape)
        print("Total %s for %s %s seconds" % (log_path, encoding.method, time.time() - start_time))

    def test_performance(self):
        encs = [SIMPLE_INDEX, BOOLEAN, FREQUENCY, COMPLEX, LAST_PAYLOAD]
        logs = ["log_cache/Sepsis Cases - Event Log.xes.gz", "log_cache/financial_log.xes.gz",
                "log_cache/BPI Challenge 2017.xes.gz"]
        for l in logs:
            for e in encs:
                self.do_test(e, l)
