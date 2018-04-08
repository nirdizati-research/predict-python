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
        job['clustering'] = 'None'
        calculate(job)

    def test_reg_linear_boolean(self):
        job = self.get_job()
        job['clustering'] = 'None'
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
        job['clustering'] = 'None'
        job['encoding'] = 'lastPayload'
        calculate(job)
