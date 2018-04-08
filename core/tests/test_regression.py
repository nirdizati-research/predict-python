from django.test import TestCase

from core.core import calculate
from core.tests.test_prepare import split_double


class TestRegression(TestCase):
    """Proof of concept tests"""

    def get_job(self):
        json = dict()
        json["clustering"] = "kmeans"
        json["split"] = split_double()
        json["method"] = "linear"
        json["encoding"] = "simpleIndex"
        json["rule"] = "remaining_time"
        json["type"] = "regression"
        return json

    def test_reg_linear(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        calculate(job)

    def test_reg_linear_split_strict_temporal(self):
        job = self.get_job()
        job['split'] = {'id': 1, 'type': 'single', 'original_log_path': 'log_cache/general_example.xes',
                        'config': {"split_type": "split_strict_temporal", "test_size": 0.30000000000000004}}
        job['clustering'] = 'noCluster'
        calculate(job)

    def test_reg_linear_boolean(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        job['encoding'] = 'boolean'
        calculate(job)

    def test_reg_randomforest(self):
        job = self.get_job()
        job['method'] = 'randomForest'
        calculate(job)

    def test_reg_lasso(self):
        job = self.get_job()
        job['method'] = 'lasso'
        calculate(job)

    # WILL NOT WORK
    def reg_lasso_complex(self):
        job = self.get_job()
        job['method'] = 'lasso'
        job['encoding'] = 'complex'
        calculate(job)

    def reg_lasso_last_payload(self):
        job = self.get_job()
        job['method'] = 'lasso'
        job['clustering'] = 'noCluster'
        job['encoding'] = 'lastPayload'
        calculate(job)
