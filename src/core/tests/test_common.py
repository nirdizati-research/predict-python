"""
common tests
"""

from django.test import TestCase

from src.core.common import get_method_config
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModelTypes
from src.utils.tests_utils import create_test_job, create_test_predictive_model


class TestCommon(TestCase):
    def test_get_method_config(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(
                prediction_type=PredictiveModelTypes.CLASSIFICATION.value,
                predictive_model_type=ClassificationMethods.RANDOM_FOREST.value
            )
        )

        method, config = get_method_config(job)

        self.assertEqual(ClassificationMethods.RANDOM_FOREST.value, method)
        self.assertEqual(create_test_predictive_model(), config)
