"""
common tests
"""

from django.test import TestCase

from src.core.common import get_method_config
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.utils.tests_utils import create_test_job, create_test_predictive_model


class TestCommon(TestCase):
    def test_get_method_config(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                prediction_method=ClassificationMethods.RANDOM_FOREST.value
            )
        )

        method, config = get_method_config(job)

        self.assertEqual(ClassificationMethods.RANDOM_FOREST.value, method)
        self.assertEqual({
            'max_depth': None,
            'max_features': 'auto',
            'n_estimators': 10,
            'random_state': 123
        }, config)
