from django.test import TestCase

from src.core.core import get_encoded_logs
from src.encoding.models import ValueEncodings
from src.jobs.models import JobTypes
from src.labelling.models import LabelTypes
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.split.models import SplitTypes, SplitOrderingMethods
from src.utils.tests_utils import create_test_split, create_test_log, create_test_encoding, create_test_job, \
    create_test_labelling, create_test_predictive_model


class TestSplitHandling(TestCase):
    """Proof of concept tests"""

    def setUp(self) -> None:
        self.log = create_test_log(
            log_name='general_example.xes',
            log_path='cache/log_cache/test_logs/general_example.xes'
        )

        self.encoding = create_test_encoding(
            prefix_length=4,
            padding=True,
            value_encoding=ValueEncodings.SIMPLE_INDEX.value
        )

        self.labelling = create_test_labelling(
            label_type=LabelTypes.NEXT_ACTIVITY.value,
        )

        self.predictive_model = create_test_predictive_model(
            predictive_model=PredictiveModels.CLASSIFICATION.value,
            prediction_method=ClassificationMethods.DECISION_TREE.value
        )

    def test_split_materialisation(self):
        split = create_test_split(
            split_type=SplitTypes.SPLIT_SINGLE.value,
            split_ordering_method=SplitOrderingMethods.SPLIT_SEQUENTIAL.value,
            test_size=0.2,
            original_log=self.log
        )
        job = create_test_job(
            split=split,
            encoding=self.encoding,
            labelling=self.labelling,
            clustering=None,
            create_models=False,
            predictive_model=self.predictive_model,
            job_type=JobTypes.PREDICTION.value,
            hyperparameter_optimizer=None,
            incremental_train=None
        )
        split_id0 = split.id
        training_df1, test_df1 = get_encoded_logs(job)
        split_id1 = job.split.id
        self.assertNotEqual(split_id0, split_id1)

    def test_split_avoid_duplication(self):
        split = create_test_split(
            split_type=SplitTypes.SPLIT_SINGLE.value,
            split_ordering_method=SplitOrderingMethods.SPLIT_SEQUENTIAL.value,
            test_size=0.2,
            original_log=self.log
        )
        job = create_test_job(
            split=split,
            encoding=self.encoding,
            labelling=self.labelling,
            clustering=None,
            create_models=False,
            predictive_model=self.predictive_model,
            job_type=JobTypes.PREDICTION.value,
            hyperparameter_optimizer=None,
            incremental_train=None
        )
        training_df1, test_df1 = get_encoded_logs(job)
        split_id1 = job.split.id
        job = create_test_job(
            split=split,
            encoding=self.encoding,
            labelling=self.labelling,
            clustering=None,
            create_models=False,
            predictive_model=self.predictive_model,
            job_type=JobTypes.PREDICTION.value,
            hyperparameter_optimizer=None,
            incremental_train=None
        )
        training_df2, test_df2 = get_encoded_logs(job)
        split_id2 = job.split.id
        self.assertEqual(split_id1, split_id2)
