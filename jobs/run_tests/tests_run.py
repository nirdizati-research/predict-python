from django.test import TestCase
from django_rq.queues import get_queue
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from core.constants import CLASSIFICATION, REGRESSION, NEXT_ACTIVITY
from jobs.models import Job, JobRun
from jobs.tasks import prediction_task, prediction
from logs.models import Log, Split
from training.models import PredModels
from training.tr_core import calculate

class JobModelTest(TestCase):
    def split_single(self):
        split = dict()
        split['type'] = 'single'
        split['original_log_path'] = 'log_cache/BPIC12_10.xes'
        return split

    def get_job(self):
        json = dict()
        json["clustering"] = "kmeans"
        json["split"] = self.split_single()
        json["method"] = "linear"
        json["encoding"] = "simpleIndex"
        json["rule"] = "remaining_time"
        json["prefix_length"] = 4
        json["threshold"] = "default"
        json["type"] = "regression"
        return json
    
    def setUp(self):
        self.config = {'key': 123,
                       'method': 'linear',
                       'encoding': 'simpleIndex',
                       'clustering': 'kmeans',
                       'prefix_length':4,
                       "rule": "remaining_time",
                       'threshold': 'default',
                       }
        log = Log.objects.create(name="BPIC12_10", path="log_cache/BPIC12_10.xes")
        log1 = Log.objects.create(name="general_example_training", path="log_cache/general_example_training.xes")
        log2 = Log.objects.create(name="general_example", path="log_cache/general_example.xes")
        log3 = Log.objects.create(name="general_example_test", path="log_cache/general_example_test.xes")
        log70 = Log.objects.create(name="BPIC12_70", path="log_cache/BPIC12_70.xes")
        JobRun.objects.create(config=self.config, log=log, type=REGRESSION)
        JobRun.objects.create(config=self.config, log=log, type='asdsd')
        JobRun.objects.create(config={}, log=log, type=REGRESSION)
    
    def test_prediction_linear(self):
        jobrun = JobRun.objects.get(id=1)
        job = self.get_job()
        calculate(job)
        model=PredModels.objects.get(id=1)
        prediction(jobrun,model)
