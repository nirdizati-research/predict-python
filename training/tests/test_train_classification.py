from django.test import TestCase
from training.tr_core import calculate
from logs.models import Log

class TestTrainClassification(TestCase):    
        
        def setUp(self):
            Log.objects.create(name="general_example_training", path="log_cache/general_example_training.xes")
            Log.objects.create(name="general_example", path="log_cache/general_example.xes")
            Log.objects.create(name="BPIC12_70", path="log_cache/BPIC12_70.xes")
        
        def split_single(self):
            split = dict()
            split['type'] = 'single'
            split['original_log_path'] = 'log_cache/general_example.xes'
            return split
    
        def get_job(self):
            json = dict()
            json["clustering"] = "kmeans"
            json["split"] = self.split_single()
            json["method"] = "randomForest"
            json["encoding"] = "simpleIndex"
            json["rule"] = "remaining_time"
            json["prefix_length"] = 1
            json["threshold"] = "default"
            json["type"] = "classification"
            return json
    
        def test_class_randomForest(self):
            job = self.get_job()
            job['clustering'] = 'noCluster'
            calculate(job)

        def class_KNN(self):
            job = self.get_job()
            job['method'] = 'KNN'
            calculate(job)
            
        def test_next_activity_DecisionTree(self):
            job = self.get_job()
            job['method'] = 'decisionTree'
            job['type'] = 'nextActivity'
            job['clustering'] = 'None'
            calculate(job)

        def test_class_complex(self):
            job = self.get_job()
            job['clustering'] = 'noCluster'
            job["encoding"] = "complex"
            calculate(job)
    
        def test_class_last_payload(self):
            job = self.get_job()
            job['clustering'] = 'noCluster'
            job["encoding"] = "lastPayload"
            calculate(job) 