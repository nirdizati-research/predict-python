"""
performance tests
"""

import time
import unittest

from django.test import TestCase

from src.core.core import calculate
from src.core.hyperopt_wrapper import calculate_hyperopt
from src.core.tests.common import split_single, add_default_config
from src.encoding.encoding_container import EncodingContainer
from src.encoding.models import ValueEncodings
from src.labelling.label_container import LabelContainer
from src.labelling.models import LabelTypes
from src.utils.tests_utils import bpi_log_filepath


@unittest.skip('performance test not needed normally')
class TestClassPerf(TestCase):
    @staticmethod
    def get_job():
        json = dict()
        json['clustering'] = 'noCluster'
        json['split'] = split_single()
        json['split']['original_log_path'] = bpi_log_filepath
        json['method'] = 'randomForest'
        json['encoding'] = EncodingContainer(ValueEncodings.BOOLEAN.value, prefix_length=20)
        json['type'] = 'classification'
        json['label'] = LabelContainer(LabelTypes.DURATION.value)
        json['incremental_train'] = {'base_model': None}
        return json

    @staticmethod
    def calculate_helper(job):
        start_time = time.time()
        calculate(job)
        print('Total for %s %s seconds' % (job['method'], time.time() - start_time))

    @staticmethod
    def calculate_helper_hyperopt(job):
        start_time = time.time()
        calculate_hyperopt(job)
        print('Total for %s %s seconds' % (job['method'], time.time() - start_time))

    def test_class_randomForest(self):
        job = self.get_job()
        add_default_config(job)
        self.calculate_helper(job)

    def test_next_activity_randomForest(self):
        job = self.get_job()
        job['label'] = LabelContainer(LabelTypes.NEXT_ACTIVITY.value)
        add_default_config(job)
        self.calculate_helper(job)

    def test_class_knn(self):
        job = self.get_job()
        job['method'] = 'knn'
        add_default_config(job)
        self.calculate_helper(job)

    def test_class_decision(self):
        job = self.get_job()
        job['method'] = 'decisionTree'
        add_default_config(job)
        self.calculate_helper(job)

    def test_class_hyperopt(self):
        job = self.get_job()
        job['label'] = LabelContainer(LabelTypes.NEXT_ACTIVITY.value)
        job['hyperopt'] = {'use_hyperopt': True, 'max_evals': 10, 'performance_metric': 'f1score'}
        add_default_config(job)
        self.calculate_helper_hyperopt(job)


@unittest.skip('performance test not needed normally')
class RegPerf(TestCase):
    @staticmethod
    def get_job():
        json = dict()
        json['clustering'] = 'noCluster'
        json['split'] = split_single()
        json['split']['original_log_path'] = bpi_log_filepath
        json['method'] = 'randomForest'
        json['encoding'] = EncodingContainer(ValueEncodings.BOOLEAN.value, prefix_length=20)
        json['prefix_length'] = 20
        json['type'] = 'regression'
        json['padding'] = 'no_padding'
        json['label'] = LabelContainer(LabelTypes.REMAINING_TIME.value)
        return json

    @staticmethod
    def calculate_helper(job):
        start_time = time.time()
        calculate(job)
        print('Total for %s %s seconds' % (job['method'], time.time() - start_time))

    @staticmethod
    def calculate_helper_hyperopt(job):
        start_time = time.time()
        calculate_hyperopt(job)
        print('Total for %s %s seconds' % (job['method'], time.time() - start_time))

    def test_reg_randomForest(self):
        job = self.get_job()
        add_default_config(job)
        self.calculate_helper(job)

    def test_reg_linear(self):
        job = self.get_job()
        job['method'] = 'linear'
        add_default_config(job)
        self.calculate_helper(job)

    def test_reg_lasso(self):
        job = self.get_job()
        job['method'] = 'lasso'
        add_default_config(job)
        self.calculate_helper(job)

    def test_reg_hyperopt(self):
        job = self.get_job()
        job['hyperopt'] = {'use_hyperopt': True, 'max_evals': 10, 'performance_metric': 'rmse'}
        add_default_config(job)
        self.calculate_helper_hyperopt(job)
