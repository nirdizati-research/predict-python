import contextlib
from os import remove

from django.test import SimpleTestCase, TestCase
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from .replayer import Replayer
from logs.models import Log

class DemoTest(TestCase):
    
    def test_demo_executions(self):
        Log.objects.create(name='general_example_test', path='log_cache/general_example_test.xes')
        replayer = Replayer(1)
        replayer.start()