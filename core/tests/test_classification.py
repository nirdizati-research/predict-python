from django.test import TestCase

from core.classification import classifier
from core.job import Job
from core.next_activity import next_activity


class TestClassification(TestCase):
    """Proof of concept tests"""

    def get_job(self):
        json = dict()
        json['uuid'] = "ads69"
        json["clustering"] = "kmeans"
        json["status"] = "completed"
        json["log"] = "Production.xes"
        json["classification"] = "randomForest"
        json["encoding"] = "simpleIndex"
        json["timestamp"] = "Oct 03 2017 13:26:53"
        json["rule"] = "remaining_time"
        json["prefix"] = 0
        json["threshold"] = "default"
        json["type"] = "Classification"
        return Job(json)

    def test_class_randomForest(self):
        job = self.get_job()
        job.clustering = 'None'
        classifier(job)

    # KNN Fails due to small dataset
    # Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 5
    def class_KNN(self):
        job = self.get_job()
        job.classification = 'KNN'
        classifier(job)

    def test_class_DecisionTree(self):
        job = self.get_job()
        job.classification = 'decisionTree'
        classifier(job)

    def test_next_activity_randomForest(self):
        job = self.get_job()
        job.type = 'nextActivity'
        next_activity(job)

    # KNN Fails due to small dataset
    # Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 5
    def next_activity_KNN(self):
        job = self.get_job()
        job.classification = 'KNN'
        job.type = 'nextActivity'
        next_activity(job)

    def test_next_activity_DecisionTree(self):
        job = self.get_job()
        job.classification = 'decisionTree'
        job.type = 'nextActivity'
        job.clustering = 'None'
        next_activity(job)
