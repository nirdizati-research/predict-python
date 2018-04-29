from django.test import TestCase

from core.core import calculate
from core.tests.test_prepare import repair_example
from encoders.label_container import LabelContainer, NEXT_ACTIVITY, THRESHOLD_CUSTOM, ATTRIBUTE_STRING, DURATION


class Labelling(TestCase):
    def get_job(self):
        json = dict()
        json["split"] = repair_example()
        json["encoding"] = "simpleIndex"
        json["prefix_length"] = 5
        json["type"] = "labelling"
        json["padding"] = 'zero_padding'
        json['label'] = LabelContainer()
        return json

    def test_remaining_time(self):
        job = self.get_job()
        result, _ = calculate(job)
        self.assertEqual(result, {'true': 529, 'false': 354})

    def test_next_activity(self):
        job = self.get_job()
        job['label'] = LabelContainer(NEXT_ACTIVITY)
        result, _ = calculate(job)
        self.assertEqual(result, {'0': 2, '3': 306, '4': 432, '5': 5, '7': 138})

    def test_remaining_custom_threshold(self):
        job = self.get_job()
        job['label'] = LabelContainer(threshold_type=THRESHOLD_CUSTOM, threshold=1600)
        result, _ = calculate(job)
        self.assertEqual(result, {'true': 444, 'false': 439})

    def test_atr_string(self):
        job = self.get_job()
        job['label'] = LabelContainer(ATTRIBUTE_STRING, attribute_name='description')
        result, _ = calculate(job)
        self.assertEqual(result, {'Simulated process instance': 883})

    def test_duration(self):
        """Trace atr, zero padding means prefix length has no effect"""
        job = self.get_job()
        job['label'] = LabelContainer(DURATION)
        result1, _ = calculate(job)
        job['prefix_length'] = 22
        result2, _ = calculate(job)
        self.assertEqual(result1, result2)
