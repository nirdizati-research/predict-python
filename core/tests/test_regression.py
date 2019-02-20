import itertools

from django.test import TestCase

from core.constants import regression_methods, NO_CLUSTER, LINEAR, REGRESSION
from core.core import calculate
from core.tests.test_prepare import split_double, add_default_config, HidePrints
from encoders.encoding_container import EncodingContainer, SIMPLE_INDEX, ZERO_PADDING, encoding_methods, \
    paddings
from encoders.label_container import LabelContainer, REMAINING_TIME, regression_labels, ATTRIBUTE_NUMBER


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

        add_default_config(json)
        return json

    def test_no_exceptions(self):
        filtered_labels = [x for x in regression_labels if
                           x != ATTRIBUTE_NUMBER]  # TODO: check how to add TRACE_NUMBER_ATTRIBUTE (test logs don't have numeric attributes
        choices = [encoding_methods, paddings, regression_methods, filtered_labels]

        job_combinations = list(itertools.product(*choices))

        for (encoding, padding, method, label) in job_combinations:
            print(encoding, padding, method, label)

            job = self.get_job(method=method, encoding_method=encoding, padding=padding, label=label)
            with HidePrints():
                calculate(job)
