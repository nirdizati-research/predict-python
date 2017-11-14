from django.test import TestCase

from core.core import calculate
from core.job import Job


class TestRegression(TestCase):
    """Proof of concept tests"""

    def get_job(self):
        json = dict()
        json['uuid'] = "ads69"
        json["clustering"] = "kmeans"
        json["status"] = "completed"
        json["log"] = "Production.xes"
        json["regression"] = "linear"
        json["encoding"] = "simpleIndex"
        json["timestamp"] = "Oct 03 2017 13:26:53"
        json["rule"] = "remaining_time"
        json["type"] = "regression"
        return Job(json)

    def test_reg_linear(self):
        job = self.get_job()
        job.clustering = 'None'
        calculate(job)

    def test_reg_randomforest(self):
        job = self.get_job()
        job.regression = 'randomForest'
        calculate(job)

    def test_reg_lasso(self):
        job = self.get_job()
        job.regression = 'lasso'
        calculate(job)
