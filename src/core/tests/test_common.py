"""
common tests
"""

from django.test import TestCase

from src.core.common import get_method_config
from src.predictive_model.regression.models import RegressionMethods


class TestCommon(TestCase):
    def test_get_method_config(self):
        job = dict()
        job['method'] = 'nn'
        job['type'] = RegressionMethods.REGRESSION
        job['regression.nn'] = 'TEST'
        job['incremental_train'] = {'base_model': None}

        method, config = get_method_config(job)

        self.assertEqual(job['method'], method)
        self.assertEqual('TEST', config)

    def test_get_method_config_exception(self):
        job = dict()
        job['method'] = 'nn'
        job['type'] = RegressionMethods.REGRESSION

        self.assertRaises(KeyError, get_method_config, job)
