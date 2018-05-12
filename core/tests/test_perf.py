# Test performance for thesis
import time
import unittest
from unittest import TestCase

from core.core import calculate
from core.hyperopt_wrapper import calculate_hyperopt
from core.tests.test_prepare import split_single, add_default_config
from encoders.encoding_container import EncodingContainer, BOOLEAN
from encoders.label_container import LabelContainer, DURATION, REMAINING_TIME, NEXT_ACTIVITY


@unittest.skip("performance test not needed normally")
class TestClassPerf(TestCase):
    def get_job(self):
        json = dict()
        json["clustering"] = "noCluster"
        json["split"] = split_single()
        json["split"]["original_log_path"] = "log_cache/BPI Challenge 2017.xes.gz"
        json["method"] = "randomForest"
        json["encoding"] = EncodingContainer(BOOLEAN, prefix_length=20)
        json["type"] = "classification"
        json['label'] = LabelContainer(DURATION)
        return json

    def calculate_helper(self, job):
        start_time = time.time()
        calculate(job)
        print("Total for %s %s seconds" % (job['method'], time.time() - start_time))

    def calculate_helper_hyperopt(self, job):
        start_time = time.time()
        calculate_hyperopt(job)
        print("Total for %s %s seconds" % (job['method'], time.time() - start_time))

    def test_class_randomForest(self):
        job = self.get_job()
        job['label'] = LabelContainer(NEXT_ACTIVITY)
        add_default_config(job)
        self.calculate_helper(job)

    def test_ne_randomForest(self):
        job = self.get_job()
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

    def test_reg_hyperopt(self):
        job = self.get_job()
        job['label'] = LabelContainer(NEXT_ACTIVITY)
        job['hyperopt'] = {'use_hyperopt': True, 'max_evals': 10, 'performance_metric': 'f1score'}
        add_default_config(job)
        self.calculate_helper_hyperopt(job)


@unittest.skip("performance test not needed normally")
class RegPerf(TestCase):
    def get_job(self):
        json = dict()
        json["clustering"] = "noCluster"
        json["split"] = split_single()
        json["split"]["original_log_path"] = "log_cache/BPI Challenge 2017.xes.gz"
        json["method"] = "randomForest"
        json["encoding"] = "boolean"
        json["prefix_length"] = 20
        json["type"] = "regression"
        json["padding"] = 'no_padding'
        json['label'] = LabelContainer(REMAINING_TIME)
        return json

    def calculate_helper(self, job):
        start_time = time.time()
        calculate(job)
        print("Total for %s %s seconds" % (job['method'], time.time() - start_time))

    def calculate_helper_hyperopt(self, job):
        start_time = time.time()
        calculate_hyperopt(job)
        print("Total for %s %s seconds" % (job['method'], time.time() - start_time))

    def test_class_randomForest(self):
        job = self.get_job()
        add_default_config(job)
        self.calculate_helper(job)

    def test_class_linear(self):
        job = self.get_job()
        job['method'] = 'linear'
        add_default_config(job)
        self.calculate_helper(job)

    def test_class_lasso(self):
        job = self.get_job()
        job['method'] = 'lasso'
        add_default_config(job)
        self.calculate_helper(job)

    def test_reg_hyperopt(self):
        job = self.get_job()
        job['hyperopt'] = {'use_hyperopt': True, 'max_evals': 10, 'performance_metric': 'rmse'}
        add_default_config(job)
        self.calculate_helper_hyperopt(job)
