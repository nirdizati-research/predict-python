from django.test import TestCase

from src.logs.models import Log
from src.utils.tests_utils import general_example_test_filepath
from .replayer import Replayer


class DemoTest(TestCase):

    @staticmethod
    def test_demo_executions():
        Log.objects.get_or_create(name='general_example_test.xes', path=general_example_test_filepath)
        replayer = Replayer(1, 13, 10)
        replayer.start()
