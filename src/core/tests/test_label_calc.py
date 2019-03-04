"""
label calculation tests
"""

import unittest

from django.test import TestCase

from src.core.core import calculate
from src.core.tests.common import repair_example
from src.encoding.encoding_container import EncodingContainer, ZERO_PADDING
from src.jobs.models import JobTypes
from src.labelling.label_container import LabelContainer
from src.labelling.models import LabelTypes, ThresholdTypes
from src.utils.tests_utils import create_test_job, create_test_encoding, create_test_labelling


class Labelling(TestCase):
    @staticmethod
    def get_job():
        return create_test_job(split=repair_example(),
                               encoding=create_test_encoding(prefix_length=5,
                                                             padding=True
                                                             ),
                               job_type=JobTypes.LABELLING.value
                               )

    @unittest.skip('needs refactoring')
    def test_remaining_time(self):
        job = self.get_job()
        result, _ = calculate(job)
        self.assertEqual(result, {'true': 529, 'false': 354})

    def test_next_activity(self):
        job = self.get_job()
        job['label'] = LabelContainer(LabelTypes.NEXT_ACTIVITY.value)
        result, _ = calculate(job)
        self.assertEqual(result, {'0': 2, 'Repair (Complex)': 306, 'Test Repair': 432, 'Inform User': 5,
                                  'Repair (Simple)': 138})

    @unittest.skip('needs refactoring')
    def test_remaining_custom_threshold(self):
        job = self.get_job()
        job['label'] = LabelContainer(threshold_type=ThresholdTypes.THRESHOLD_CUSTOM.value, threshold=1600)
        result, _ = calculate(job)
        self.assertEqual(result, {'true': 444, 'false': 439})

    def test_atr_string(self):
        job = self.get_job()
        job.labelling = create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value,
                                              attribute_name='description')
        job.save()
        result, _ = calculate(job)
        self.assertEqual(result, {'Simulated process instance': 883})

    def test_duration(self):
        """Trace atr, zero padding means prefix length has no effect"""
        job = self.get_job()
        job['label'] = LabelContainer(LabelTypes.DURATION.value)
        result1, _ = calculate(job)
        job['encoding'] = EncodingContainer(prefix_length=22, padding=ZERO_PADDING)
        result2, _ = calculate(job)
        self.assertEqual(result1, result2)
