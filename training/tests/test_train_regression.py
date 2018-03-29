from django.test import TestCase
from training.tr_core import calculate
from logs.models import Log
from core.constants import KMEANS, NO_CLUSTER

class TestTrainRegression(TestCase):
        
        def setUp(self):
            Log.objects.create(name="general_example_training", path="log_cache/general_example_training.xes")
            Log.objects.create(name="general_example", path="log_cache/general_example.xes")
            Log.objects.create(name="BPIC12_70", path="log_cache/BPIC12_70.xes")
        
        def split_single(self):
            split = dict()
            split['type'] = 'single'
            split['original_log_path'] = 'log_cache/general_example_training.xes'
            return split
    
        def get_job(self):
            json = dict()
            json["clustering"] = KMEANS
            json["split"] = self.split_single()
            json["method"] = "linear"
            json["prefix_length"] = 4
            json["encoding"] = "simpleIndex"
            json["rule"] = "remaining_time"
            json["type"] = "regression"
            return json
    
        def test_reg_linear(self):
            job = self.get_job()
            job['clustering'] = 'None'
            calculate(job, redo=True)
            
        def test_reg_linear_boolean(self):
            job = self.get_job()
            job['clustering'] = 'None'
            job['encoding'] = 'boolean'
            calculate(job, redo=True)

        def test_reg_randomforest(self):
            job = self.get_job()
            job['method'] = 'randomForest'
            calculate(job, redo=True)

        def test_reg_lasso(self):
            job = self.get_job()
            job['method'] = 'lasso'
            calculate(job, redo=True)
            
        