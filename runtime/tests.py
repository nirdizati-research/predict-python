from django.test import TestCase

from logs.models import Log
from .replayer import Replayer


class DemoTest(TestCase):

    @staticmethod
    def test_demo_executions():
        Log.objects.get_or_create(name='general_example_test.xes', path='log_cache/general_example_test.xes')
        replayer = Replayer(1, 13, 10)
        replayer.start()
