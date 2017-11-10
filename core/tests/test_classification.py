from django.test import TestCase

from core.classification import classifier
from core.job import Job
from core.next_activity import next_activity


class TestClassification(TestCase):
    def get_job(self):
        json = dict()
        json['uuid'] = "ads69"
        json["clustering"] = "kmeans"
        json["status"] = "completed"
        json["run"] = "RandomForest_simpleIndex_None"
        json["log"] = "Production.xes"
        json["classification"] = "RandomForest"
        json["encoding"] = "simpleIndex"
        json["timestamp"] = "Oct 03 2017 13:26:53"
        json["rule"] = "remaining_time"
        json["prefix"] = 0
        json["threshold"] = "default"
        json["type"] = "Classification"
        return Job(json)

    def test_job(self):
        job = self.get_job()
        classifier(job)

    def test_next_activity_job(self):
        job = self.get_job()
        next_activity(job)
