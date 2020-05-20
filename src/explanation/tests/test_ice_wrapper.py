from django.test import TestCase

from src.core.core import get_encoded_logs
from src.encoding.models import ValueEncodings
from src.explanation.ice_wrapper import explain
from src.explanation.models import Explanation, ExplanationTypes
from src.jobs.models import JobTypes
from src.jobs.tasks import prediction_task
from src.labelling.models import LabelTypes
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.split.models import SplitTypes, SplitOrderingMethods
from src.utils.tests_utils import create_test_split, create_test_log, create_test_predictive_model, create_test_job, \
    create_test_encoding, create_test_labelling


class TestICEWrapper(TestCase):
    """Proof of concept tests"""

    def test_explain(self):
        split = create_test_split(
            split_type=SplitTypes.SPLIT_DOUBLE.value,
            split_ordering_method=SplitOrderingMethods.SPLIT_SEQUENTIAL.value,
            test_size=0.2,
            original_log=None,
            train_log=create_test_log(
                log_name='general_example.xes',
                log_path='cache/log_cache/test_logs/train_explainability.xes'
            ),
            test_log=create_test_log(
                log_name='general_example_train.xes',
                log_path='cache/log_cache/test_logs/test_explainability.xes'
            )
        )

        predictive_model = create_test_predictive_model(
            predictive_model=PredictiveModels.CLASSIFICATION.value,
            prediction_method=ClassificationMethods.DECISION_TREE.value
        )

        job = create_test_job(
            split=split,
            encoding=create_test_encoding(
                prefix_length=4,
                padding=True,
                value_encoding=ValueEncodings.SIMPLE_INDEX.value
            ),
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value, attribute_name='label'),
            clustering=None,
            create_models=True,
            predictive_model=predictive_model,
            job_type=JobTypes.PREDICTION.value,
            hyperparameter_optimizer=None,
            incremental_train=None
        )

        prediction_task(job.id, do_publish_result=False)
        job.refresh_from_db()

        exp = Explanation.objects.get_or_create(
            type=ExplanationTypes.ICE.value,
            split=split,
            predictive_model=predictive_model,
            job=job,
            results={}
        )[0]
        training_df_old, test_df_old = get_encoded_logs(job)

        explanation_target = 'prefix_2'

        explanation = explain(exp, training_df_old, test_df_old, explanation_target)

        expected = [
            {'value': 'Contact Hospital', 'label': 1.2962962962962963, 'count': 351},
            {'value': 'Create Questionnaire', 'label': 1.5526992287917738, 'count': 1167},
            {'value': 'High Insurance Check', 'label': 1.2667660208643816, 'count': 671}
        ]

        self.assertEqual(expected, explanation)
