from django.test import SimpleTestCase
from opyenxes.data_in.XesXmlParser import XesXmlParser
from .log_service import events_by_date


class LogTest(SimpleTestCase):

    def test_events_by_date(self):
        with open("log_cache/general_example.xes") as file:
            logs = XesXmlParser().parse(file)
        result = events_by_date(logs)
        self.assertEqual(18, len(result.keys()))
        self.assertEqual(4, result['2011-01-08'])
