from django.test import TestCase

from core.hyperopt_wrapper import calculate_hyperopt
from core.tests.test_prepare import add_default_config, repair_example


class TestHyperopt(TestCase):
    """Proof of concept tests"""

    def get_job(self):
        json = dict()
        json["clustering"] = "kmeans"
        json["split"] = repair_example()
        json["method"] = "randomForest"
        json["encoding"] = "simpleIndex"
        json["rule"] = "remaining_time"
        json["prefix_length"] = 8
        json["threshold"] = "default"
        json["type"] = "classification"
        json["padding"] = 'zero_padding'
        return json

    def test_class_randomForest(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        add_default_config(job)
        results, config = calculate_hyperopt(job, max_evals=5)
        print("Best hyperopt config")
        print(config)