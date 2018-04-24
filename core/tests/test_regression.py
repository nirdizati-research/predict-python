from django.test import TestCase

from core.constants import ZERO_PADDING
from core.core import calculate
from core.tests.test_prepare import split_double, add_default_config
from encoders.label_container import LabelContainer


class TestRegression(TestCase):
    """Proof of concept tests"""

    def get_job(self):
        json = dict()
        json["clustering"] = "kmeans"
        json["split"] = split_double()
        json["method"] = "linear"
        json["encoding"] = "simpleIndex"
        json["label"] = LabelContainer()
        json["type"] = "regression"
        json["padding"] = ZERO_PADDING
        return json

    def test_reg_linear(self):
        job = self.get_job()
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
        job['encoding'] = 'boolean'
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

    def test_reg_lasso_no_elapsed_time(self):
        job = self.get_job()
        job['method'] = 'lasso'
        job['add_elapsed_time'] = False
        add_default_config(job)
        calculate(job)

    # WILL NOT WORK
    def reg_lasso_complex(self):
        job = self.get_job()
        job['method'] = 'lasso'
        job['encoding'] = 'complex'
        add_default_config(job)
        calculate(job)

    def reg_lasso_last_payload(self):
        job = self.get_job()
        job['method'] = 'lasso'
        job['clustering'] = 'noCluster'
        job['encoding'] = 'lastPayload'
        add_default_config(job)
        calculate(job)
