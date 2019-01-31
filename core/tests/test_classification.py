from django.test import TestCase

from core.core import calculate
from core.tests.test_prepare import split_double, add_default_config
from encoders.encoding_container import EncodingContainer, COMPLEX, LAST_PAYLOAD, ZERO_PADDING
from encoders.label_container import LabelContainer, NEXT_ACTIVITY, ATTRIBUTE_STRING, THRESHOLD_CUSTOM
from jobs.job_creator import _kmeans


class TestClassification(TestCase):
    """Proof of concept tests"""

    @staticmethod
    def results():
        return {'f1score': 0.66666666666666663, 'acc': 0.5, 'auc': 0.16666666666666666, 'false_negative': 0,
                'false_positive': 1, 'true_positive': 1, 'true_negative': 0, 'precision': 1.0, 'recall': 0.5}

    @staticmethod
    def results2():
        return {'f1score': 0.6666666666666666, 'auc': 0.5, 'acc': 0.5, 'false_negative': 0, 'false_positive': 1,
                'true_positive': 1, 'true_negative': 0, 'precision': 0.5, 'recall': 1.0}

    @staticmethod
    def results3():
        return {'f1score': 0.3333333333333333, 'acc': 0.5, 'auc': 0, 'precision': 0.25, 'recall': 0.5}

    @staticmethod
    def get_job():
        json = dict()
        json["clustering"] = "kmeans"
        json["split"] = split_double()
        json["method"] = "randomForest"
        json["encoding"] = EncodingContainer()
        json["type"] = "classification"
        json['label'] = LabelContainer(add_elapsed_time=True)
        return json

    def test_class_randomForest(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        add_default_config(job)
        result, _ = calculate(job)
        self.assertDictEqual(result, self.results2())

    def test_class_randomForest_p4(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        job["prefix_length"] = 4
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_class_KNN(self):
        job = self.get_job()
        job['method'] = 'knn'
        job['clustering'] = 'noCluster'
        job['classification.knn'] = {'n_neighbors': 3}
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_class_DecisionTree(self):
        job = self.get_job()
        job['method'] = 'decisionTree'
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_class_xgboost(self):
        job = self.get_job()
        job['method'] = 'xgboost'
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_next_activity_randomForest(self):
        job = self.get_job()
        job['label'] = LabelContainer(NEXT_ACTIVITY)
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_next_activity_KNN(self):
        job = self.get_job()
        job['method'] = 'knn'
        job['label'] = LabelContainer(NEXT_ACTIVITY)
        job['classification.knn'] = {'n_neighbors': 3}
        job['kmeans'] = _kmeans()
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_next_activity_xgboost(self):
        job = self.get_job()
        job['method'] = 'xgboost'
        job['label'] = LabelContainer(NEXT_ACTIVITY)
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_attribute_string_knn(self):
        job = self.get_job()
        job['method'] = 'knn'
        job['label'] = LabelContainer(ATTRIBUTE_STRING, attribute_name='creator')
        job['classification.knn'] = {'n_neighbors': 3}
        job['kmeans'] = _kmeans()
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_next_activity_DecisionTree(self):
        job = self.get_job()
        job['method'] = 'decisionTree'
        job['label'] = LabelContainer(NEXT_ACTIVITY)
        job['clustering'] = 'noCluster'
        add_default_config(job)
        result, _ = calculate(job)
        self.assertDictEqual(result, self.results3())

    def test_class_complex(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        job["encoding"] = EncodingContainer(COMPLEX)
        add_default_config(job)
        result, _ = calculate(job)
        # it works, but results are unreliable

    def test_class_complex_zero_padding(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        job["encoding"] = EncodingContainer(COMPLEX, prefix_length=8, padding=ZERO_PADDING)
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)
        # it works, but results are unreliable

    def test_class_last_payload(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        job["encoding"] = EncodingContainer(LAST_PAYLOAD)
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)
        # it works, but results are unreliable

    def test_class_last_payload_custom_threshold(self):
        job = self.get_job()
        job['clustering'] = 'noCluster'
        job["encoding"] = EncodingContainer(LAST_PAYLOAD, prefix_length=5)
        job['label'] = LabelContainer(threshold_type=THRESHOLD_CUSTOM, threshold=50)
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)
        # it works, but results are unreliable
