import time
from unittest import TestCase

from core.constants import REGRESSION
from encoders.common import encode_label_log, BOOLEAN
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
