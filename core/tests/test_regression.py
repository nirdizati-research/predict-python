import unittest

from django.test import TestCase

from core.core import calculate
from core.tests.test_prepare import split_double, add_default_config, split_single
from encoders.encoding_container import EncodingContainer, SIMPLE_INDEX, ZERO_PADDING, BOOLEAN, COMPLEX, LAST_PAYLOAD
from encoders.label_container import LabelContainer, REMAINING_TIME


class TestRegression(TestCase):
    """Proof of concept tests"""

    @staticmethod
    def get_job():
        json = dict()
        json["clustering"] = "kmeans"
        json["split"] = split_double()
        json["method"] = "linear"
        json["encoding"] = EncodingContainer(SIMPLE_INDEX, padding=ZERO_PADDING)
        json["label"] = LabelContainer()
        json["type"] = "regression"
        return json

    def test_reg_linear(self):
        job = self.get_job()
        job["encoding"] = EncodingContainer(SIMPLE_INDEX, padding=ZERO_PADDING, prefix_length=13)
        job['clustering'] = 'noCluster'
        add_default_config(job)
        calculate(job)

    def test_reg_linear_split_strict_temporal(self):
        job = self.get_job()
        job['split'] = {'id': 1, 'type': 'single', 'original_log_path': 'log_cache/general_example.xes',
                        'config': {"split_type": "split_strict_temporal", "test_size": 0.30000000000000004}}
        job['clustering'] = 'noCluster'
        add_default_config(job)
        calculate(job)

    def test_reg_linear_boolean(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        job['encoding'] = EncodingContainer(BOOLEAN, padding=ZERO_PADDING)
        add_default_config(job)
        calculate(job)

    def test_reg_randomforest(self):
        job = self.get_job()
        job['method'] = 'randomForest'
        add_default_config(job)
        calculate(job)

    def test_reg_lasso(self):
        job = self.get_job()
        job['method'] = 'lasso'
        add_default_config(job)
        calculate(job)

    def test_reg_xgboost(self):
        job = self.get_job()
        job['method'] = 'xgboost'
        add_default_config(job)
        calculate(job)

    def test_reg_lasso_no_elapsed_time(self):
        job = self.get_job()
        job['method'] = 'lasso'
        job['add_elapsed_time'] = False
        add_default_config(job)
        calculate(job)

    # WILL NOT WORK
    def test_reg_lasso_complex(self):
        job = self.get_job()
        job['method'] = 'lasso'
        job['encoding'] = EncodingContainer(COMPLEX, padding=ZERO_PADDING)
        add_default_config(job)
        calculate(job)

    def test_reg_lasso_last_payload(self):
        job = self.get_job()
        job['method'] = 'lasso'
        job['clustering'] = 'noCluster'
        job['encoding'] = EncodingContainer(LAST_PAYLOAD, padding=ZERO_PADDING)
        add_default_config(job)
        calculate(job)


@unittest.skip("evaluation test not needed normally")
class TestEvaluation(TestCase):
    @property
    def get_job(self):
        json = dict()
        json["clustering"] = "noCluster"
        json["split"] = split_single()
        json['split']['original_log_path'] = 'log_cache/Sepsis Cases - Event Log.xes'
        json["method"] = "lasso"
        json["encoding"] = EncodingContainer(COMPLEX, padding=ZERO_PADDING, prefix_length=16)
        json["label"] = LabelContainer(REMAINING_TIME)
        json["type"] = "regression"
        return json

    def test_reg_lasso(self):
        job = self.get_job
        add_default_config(job)
        calculate(job)

    def test_reg_intercase(self):
        print('interase')
        job = self.get_job
        job["label"] = LabelContainer(REMAINING_TIME, add_resources_used=True, add_new_traces=True,
                                      add_executed_events=True)
        add_default_config(job)
        calculate(job)
