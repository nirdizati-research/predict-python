"""
classification tests
"""

import itertools
import unittest

from django.test import TestCase

from src.clustering.models import ClusteringMethods
from src.core.core import calculate
from src.encoding.models import ValueEncodings
from src.labelling.models import LabelTypes, ThresholdTypes
from src.predictive_model.classification.models import ClassificationMethods
from src.utils.tests_utils import create_test_job, create_test_encoding, create_test_labelling, \
    create_test_predictive_model, create_test_clustering


class TestClassification(TestCase):
    @unittest.skip('needs refactoring')
    def test_no_exceptions(self):
        filtered_labels = [enum.value for enum in LabelTypes]

        filtered_classification_methods = [enum.value for enum in ClassificationMethods]

        filtered_encoding_methods = [enum.value for enum in ValueEncodings]

        filtered_padding = [True, False]

        choices = [filtered_encoding_methods, filtered_padding, filtered_classification_methods, filtered_labels]

        job_combinations = list(itertools.product(*choices))

        for (encoding, padding, method, label) in job_combinations:
            print(encoding, padding, method, label)

            if method == 'nn' and (padding == False or label == LabelTypes.ATTRIBUTE_STRING.value):
                pass
            job = create_test_job(
                predictive_model=create_test_predictive_model(prediction_method=method),
                encoding=create_test_encoding(value_encoding=encoding, padding=padding),
                labelling=create_test_labelling(label_type=label)
            )
            # with HidePrints():
            calculate(job)

    @staticmethod
    def results():
        return {'f1score': 0.66666666666666663, 'acc': 0.5, 'auc': 0.16666666666666666, 'false_negative': 0,
                'false_positive': 1, 'true_positive': 1, 'true_negative': 0, 'precision': 1.0, 'recall': 0.5}

    @staticmethod
    def results2():
        return {'f1score': 0.0, 'acc': 0.0, 'true_positive': 0, 'true_negative': 0, 'false_negative': 2,
                'false_positive': 0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0}

    @staticmethod
    def results3():
        return {'f1score': 0.3333333333333333, 'acc': 0.5, 'true_positive': 0, 'true_negative': 1, 'false_negative': 1,
                'false_positive': 0, 'precision': 0.25, 'recall': 0.5, 'auc': 0.5}

    def test_class_knn(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.KNN.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'true_positive': 0, 'true_negative': 0, 'false_negative': 2, 'false_positive': 0, 'auc': 0.0})

    def test_class_randomForest(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.RANDOM_FOREST.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, self.results2())

    def test_class_decisionTree(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.DECISION_TREE.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'true_positive': 0, 'true_negative': 0, 'false_negative': 2, 'false_positive': 0, 'auc': 0.0})

    def test_class_xgboost(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.XGBOOST.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'true_positive': 0, 'true_negative': 0, 'false_negative': 2, 'false_positive': 0, 'auc': 0.0})

    def test_class_nb(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'true_positive': 0, 'true_negative': 0, 'false_negative': 2, 'false_positive': 0, 'auc': 0.0})

    def test_class_hoeff(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.HOEFFDING_TREE.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'true_positive': 0, 'true_negative': 0, 'false_negative': 2, 'false_positive': 0, 'auc': 0.0})

    def test_class_ada(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.ADAPTIVE_TREE.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'true_positive': 0, 'true_negative': 0, 'false_negative': 2, 'false_positive': 0, 'auc': 0.0})

    def test_class_sgd(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.SGDCLASSIFIER.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'true_positive': 0, 'true_negative': 0, 'false_negative': 2, 'false_positive': 0, 'auc': 0.0})

    def test_class_perc(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.PERCEPTRON.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'true_positive': 0, 'true_negative': 0, 'false_negative': 2, 'false_positive': 0, 'auc': 0.0})

    def test_next_activity_DecisionTree(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.DECISION_TREE.value),
            labelling=create_test_labelling(label_type=LabelTypes.NEXT_ACTIVITY.value),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, self.results3())

    def test_class_nn(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.NN.value),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                            attribute_name='concept:name'),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 1.0, 'acc': 1.0, 'precision': 1.0, 'recall': 1.0, 'true_positive': 2,
                                      'true_negative': 0, 'false_negative': 0, 'false_positive': 0, 'auc': 0.0})

    def test_class_nn_binary(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(prediction_method=ClassificationMethods.NN.value),
            labelling=create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value,
                                            threshold_type=ThresholdTypes.THRESHOLD_MEAN.value),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'f1score': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'true_positive': 0,
                                      'true_negative': 0, 'false_negative': 2, 'false_positive': 0, 'auc': 0.0})

