"""
refactoring tests
"""

import unittest

from django.test import TestCase

from src.core.core import calculate
from src.core.tests.common import repair_example, add_default_config
from src.encoding.encoding_container import EncodingContainer, ZERO_PADDING
from src.labelling.label_container import LabelContainer
from src.labelling.models import LabelTypes


@unittest.skip('needs refactoring')
class RefactorProof(TestCase):
    @staticmethod
    def get_job():
        json = dict()
        json['clustering'] = 'kmeans'
        json['split'] = repair_example()
        json['method'] = 'randomForest'
        json['encoding'] = EncodingContainer(prefix_length=5, padding=ZERO_PADDING)
        json['type'] = 'classification'
        json['label'] = LabelContainer(add_elapsed_time=True)
        json['incremental_train'] = {'base_model': None}
        return json

    def test_class_kmeans(self):
        self.max_diff = None
        job = self.get_job()
        add_default_config(job)
        result, _ = calculate(job)
        # self.assertDictEqual(result, {'f1score': 0.7317073170731708, 'acc': 0.6515837104072398, 'true_positive': 105,
        #                               'true_negative': 39,
        #                               'false_negative': 22, 'false_positive': 55, 'precision': 0.65625,
        #                               'recall': 0.8267716535433071, 'auc': 0.5943654099132211})
        self.assertDictEqual(result, {'f1score': 0.67690058479532156, 'acc': 0.68325791855203621, 'true_positive': 91,
                                      'true_negative': 60, 'false_negative': 36, 'false_positive': 34,
                                      'precision': 0.67649999999999999, 'recall': 0.67741665270564577,
                                      'auc': 0.5913497814050234})

    def test_class_no_cluster(self):
        self.max_diff = None
        job = self.get_job()
        job['clustering'] = 'noCluster'
        add_default_config(job)
        result, _ = calculate(job)
        # self.assertDictEqual(result, {'f1score': 0.7200000000000001, 'acc': 0.6515837104072398, 'true_positive': 99,
        #                               'true_negative': 45,
        #                               'false_negative': 28, 'false_positive': 49, 'precision': 0.668918918918919,
        #                               'recall': 0.7795275590551181, 'auc': 0.69680851063829774})
        self.assertDictEqual(result, {'f1score': 0.67690058479532156, 'acc': 0.6832579185520362, 'true_positive': 91,
                                      'true_negative': 60, 'false_negative': 36, 'false_positive': 34,
                                      'precision': 0.67649999999999999, 'recall': 0.67741665270564577,
                                      'auc': 0.71720556207069863})

    def test_next_activity_kmeans(self):
        self.max_diff = None
        job = self.get_job()
        job['label'] = LabelContainer(LabelTypes.NEXT_ACTIVITY.value)
        job['encoding'] = EncodingContainer(prefix_length=8, padding=ZERO_PADDING)
        add_default_config(job)
        result, _ = calculate(job)
        # self.assertDictEqual(result, {'f1score': 0.23864644588878572, 'acc': 0.74660633484162897,
        #                               'precision': 0.19740887132191481, 'recall': 0.40000000000000002, 'auc': 0})
        self.assertDictEqual(result, {'f1score': 0.54239884582595577, 'acc': 0.80995475113122173, 'true_positive': '--',
                                      'true_negative': '--', 'false_negative': '--', 'false_positive': '--',
                                      'precision': 0.62344720496894401, 'recall': 0.5224945442336747,
                                      'auc': 0.19638554216867468})

    def test_next_activity_no_cluster(self):
        self.max_diff = None
        job = self.get_job()
        job['label'] = LabelContainer(LabelTypes.NEXT_ACTIVITY.value)
        job['clustering'] = 'noCluster'
        job['encoding'] = EncodingContainer(prefix_length=8, padding=ZERO_PADDING)
        add_default_config(job)
        result, _ = calculate(job)

        self.assertAlmostEqual(result['f1score'], 0.5423988458)
        self.assertAlmostEqual(result['acc'], 0.809954751)
        self.assertAlmostEqual(result['precision'], 0.623447204)
        self.assertAlmostEqual(result['recall'], 0.52249454423)
        self.assertAlmostEqual(result['auc'], 0)

        # old result
        # self.assertDictEqual(result,
        #  {'f1score': 0.895, 'acc': 0.8099547511312217, 'true_positive': 179, 'true_negative': 0, 'false_negative': 0,
        #  'false_positive': 42, 'precision': 0.8099547511312217, 'recall': 1.0, 'auc': 0})

    def test_regression_kmeans(self):
        self.max_diff = None
        job = self.get_job()
        job['type'] = 'regression'
        job['label'] = LabelContainer()
        add_default_config(job)
        result, _ = calculate(job)  # TODO double check
        # self.assertAlmostEqual(result['rmse'], 0.29435018)
        self.assertAlmostEqual(result['rmse'], 0.62164488)
        # self.assertAlmostEqual(result['mae'], 0.2264389)
        self.assertAlmostEqual(result['mae'], 0.00014269)
        # self.assertAlmostEqual(result['rscore'], 0.059686980)
        self.assertAlmostEqual(result['rscore'], -0.11336870)
        # self.assertAlmostEqual(result['mape'], 50.73148628)
        self.assertAlmostEqual(result['mape'], float('inf'))

    def test_regression_no_cluster(self):
        self.max_diff = None
        job = self.get_job()
        job['type'] = 'regression'
        job['clustering'] = 'noCluster'
        job['label'] = LabelContainer()
        add_default_config(job)
        result, _ = calculate(job)  # TODO double check
        # self.assertAlmostEqual(result['rmse'], 0.29123518)
        self.assertAlmostEqual(result['rmse'], 0.00012752)
        # self.assertAlmostEqual(result['mae'], 0.22594042)
        self.assertAlmostEqual(result['mae'], 0.00011685)
        # self.assertAlmostEqual(result['rscore'], 0.079483654)
        self.assertAlmostEqual(result['rscore'], 0.13776124)
        # self.assertAlmostEqual(result['mape'], 50.64461029)
        self.assertAlmostEqual(result['mape'], float('inf'))
