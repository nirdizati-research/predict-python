"""
regression tests
"""

import unittest

from django.test import TestCase

from src.clustering.models import ClusteringMethods
from src.core.tests.common import split_double, add_default_config
from src.encoding.encoding_container import EncodingContainer, ZERO_PADDING
from src.encoding.models import ValueEncodings
from src.labelling.label_container import LabelContainer
from src.labelling.models import LabelTypes
from src.predictive_model.models import PredictiveModels
from src.predictive_model.regression.models import RegressionMethods


class TestRegression(TestCase):
    @staticmethod
    def get_job(method=RegressionMethods.LINEAR.value, encoding_method=ValueEncodings.SIMPLE_INDEX.value,
                padding=ZERO_PADDING, label=LabelTypes.REMAINING_TIME.value,
                add_elapsed_time=False):
        json = dict()
        json['clustering'] = ClusteringMethods.NO_CLUSTER.value
        json['split'] = split_double()
        json['method'] = method
        json['encoding'] = EncodingContainer(encoding_method, padding=padding, prefix_length=4)
        json['labelling'] = LabelContainer(label)
        json['add_elapsed_time'] = add_elapsed_time
        json['type'] = PredictiveModels.REGRESSION.value
        json['incremental_train'] = {'base_model': None}

        add_default_config(json)
        return json

    @unittest.skip('needs refactoring')
    def test_no_exceptions(self):
        # filtered_labels = [x for x in REGRESSION_LABELS if
        #                    x != ATTRIBUTE_NUMBER]
        # # TODO: check how to add TRACE_NUMBER_ATTRIBUTE (test logs don't have numeric attributes)
        # choices = [ENCODING_METHODS, PADDINGS, REGRESSION_METHODS, filtered_labels]
        #
        # job_combinations = list(itertools.product(*choices))
        #
        # for (encoding, padding, method, label) in job_combinations:
        #     print(encoding, padding, method, label)
        #
        #     if method == 'nn' and padding == NO_PADDING:
        #         pass
        #
        #     job = self.get_job(method=method, encoding_method=encoding, padding=padding, label=label)
        #     with HidePrints():
        #         calculate(job)
        pass
