from django.test import TestCase

from core.constants import MULTINOMIAL_NAIVE_BAYES, ADAPTIVE_TREE, HOEFFDING_TREE, \
    SGDCLASSIFIER, PERCEPTRON, DECISION_TREE, XGBOOST, KNN, KMEANS
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
        return {'f1score': 0.3333333333333333, 'acc': 0.5, 'true_positive': 1, 'true_negative': 0, 'false_negative': 0,
                'false_positive': 1, 'precision': 0.25, 'recall': 0.5, 'auc': 0.5}

    @staticmethod
    def results3():
        return {'f1score': 0.3333333333333333, 'acc': 0.5, 'true_positive': 0, 'true_negative': 1, 'false_negative': 1,
                'false_positive': 0, 'precision': 0.25, 'recall': 0.5, 'auc': 0.5}

    @staticmethod
    def get_job():
        json = dict()
        json["clustering"] = KMEANS
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
        job['method'] = DECISION_TREE
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_class_xgboost(self):
        job = self.get_job()
        job['method'] = XGBOOST
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_class_mnb(self):
        job = self.get_job()
        job['method'] = MULTINOMIAL_NAIVE_BAYES
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_class_ada(self):
        job = self.get_job()
        job['method'] = ADAPTIVE_TREE
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_class_hoeff(self):
        job = self.get_job()
        job['method'] = HOEFFDING_TREE
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_class_sgdc(self):
        job = self.get_job()
        job['method'] = SGDCLASSIFIER
        job['classification.' + SGDCLASSIFIER] = dict()
        job['classification.' + SGDCLASSIFIER]['loss'] = 'log'
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_class_perceptron(self):
        job = self.get_job()
        job['method'] = PERCEPTRON
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
        job['method'] = KNN
        job['label'] = LabelContainer(NEXT_ACTIVITY)
        job['classification.knn'] = {'n_neighbors': 3}
        job['kmeans'] = _kmeans()
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_next_activity_xgboost(self):
        job = self.get_job()
        job['method'] = XGBOOST
        job['label'] = LabelContainer(NEXT_ACTIVITY)
        add_default_config(job)
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_attribute_string_knn(self):
        job = self.get_job()
        job['method'] = KNN
        job['label'] = LabelContainer(ATTRIBUTE_STRING, attribute_name='creator')
        job['classification.knn'] = {'n_neighbors': 3}
        job['kmeans'] = _kmeans()
        result, _ = calculate(job)
        self.assertIsNotNone(result)

    def test_next_activity_DecisionTree(self):
        job = self.get_job()
        job['method'] = DECISION_TREE
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
