"""
common tests
"""

from django.test import TestCase

from src.core.common import get_method_config
from src.predictive_model.models import PredictiveModelTypes


class TestCommon(TestCase):
    def test_get_method_config(self):
        job = dict()
        job['method'] = 'nn'
        job['type'] = PredictiveModelTypes.REGRESSION.value
        job['regression.nn'] = 'TEST'
        job['incremental_train'] = {'base_model': None}

        method, config = get_method_config(job)

        self.assertEqual(job['method'], method)
        self.assertEqual('TEST', config)

    def test_get_method_config_exception(self):
        job = dict()
        job['method'] = 'nn'
        job['type'] = PredictiveModelTypes.REGRESSION.value

        self.assertRaises(KeyError, get_method_config, job)
