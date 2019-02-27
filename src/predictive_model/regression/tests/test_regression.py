"""
regression tests
"""

import itertools

from django.test import TestCase

from src.core.constants import REGRESSION_METHODS, NO_CLUSTER, LINEAR, REGRESSION
from src.core.core import calculate
from src.core.tests.common import split_double, add_default_config, HidePrints
from src.encoding.encoding_container import EncodingContainer, SIMPLE_INDEX, ZERO_PADDING, ENCODING_METHODS, \
    PADDINGS, NO_PADDING
from src.labelling.label_container import LabelContainer, REMAINING_TIME, REGRESSION_LABELS, ATTRIBUTE_NUMBER


class TestRegression(TestCase):
    @staticmethod
    def get_job(method=LINEAR, encoding_method=SIMPLE_INDEX, padding=ZERO_PADDING, label=REMAINING_TIME,
                add_elapsed_time=False):
        json = dict()
        json['clustering'] = NO_CLUSTER
        json['split'] = split_double()
        json['method'] = method
        json['encoding'] = EncodingContainer(encoding_method, padding=padding, prefix_length=4)
        json['label'] = LabelContainer(label)
        json['add_elapsed_time'] = add_elapsed_time
        json['type'] = REGRESSION
        json['incremental_train'] = {'base_model': None}

        add_default_config(json)
        return json

    def test_no_exceptions(self):
        filtered_labels = [x for x in REGRESSION_LABELS if
                           x != ATTRIBUTE_NUMBER]
        # TODO: check how to add TRACE_NUMBER_ATTRIBUTE (test logs don't have numeric attributes)
        choices = [ENCODING_METHODS, PADDINGS, REGRESSION_METHODS, filtered_labels]

        job_combinations = list(itertools.product(*choices))

        for (encoding, padding, method, label) in job_combinations:
            print(encoding, padding, method, label)

            if method == 'nn' and padding == NO_PADDING:
                pass

            job = self.get_job(method=method, encoding_method=encoding, padding=padding, label=label)
            with HidePrints():
                calculate(job)
