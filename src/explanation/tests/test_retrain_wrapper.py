
from django.test import TestCase

from src.core.core import get_encoded_logs
from src.encoding.models import ValueEncodings
from src.explanation.models import Explanation, ExplanationTypes
from src.explanation.retrain_wrapper import randomise_features, save_randomised_set, explain
from src.jobs.models import JobTypes
from src.jobs.tasks import prediction_task
from src.labelling.models import LabelTypes
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.split.models import SplitTypes, SplitOrderingMethods
from src.utils.tests_utils import create_test_log, create_test_split, create_test_job, create_test_predictive_model, \
    create_test_labelling, create_test_encoding


class TestRetrainWrapper(TestCase):
    """Proof of concept tests"""

    def setUp(self) -> None:

        split = create_test_split(
            split_type=SplitTypes.SPLIT_DOUBLE.value,
            split_ordering_method=SplitOrderingMethods.SPLIT_SEQUENTIAL.value,
            test_size=0.2,
            original_log=None,
            train_log=create_test_log(
                log_name='train_explainability.xes',
                log_path='cache/log_cache/test_logs/train_explainability.xes'
            ),
            test_log=create_test_log(
                log_name='test_explainability.xes',
                log_path='cache/log_cache/test_logs/test_explainability.xes'
            )
        )

        predictive_model = create_test_predictive_model(
            predictive_model=PredictiveModels.CLASSIFICATION.value,
            prediction_method=ClassificationMethods.DECISION_TREE.value
        )

        self.job = create_test_job(
            split=split,
            encoding=create_test_encoding(
                prefix_length=4,
                padding=True,
                value_encoding=ValueEncodings.SIMPLE_INDEX.value
            ),
            labelling=create_test_labelling(
                label_type=LabelTypes.ATTRIBUTE_STRING.value,
                attribute_name='label'
            ),
            clustering=None,
            create_models=False,
            predictive_model=predictive_model,
            job_type=JobTypes.PREDICTION.value,
            hyperparameter_optimizer=None,
            incremental_train=None
        )

        prediction_task(self.job.id, do_publish_result=False)
        self.job.refresh_from_db()

        self.exp = Explanation.objects.get_or_create(
            type=ExplanationTypes.RETRAIN.value,
            split=split,
            predictive_model=predictive_model,
            job=self.job,
            results={}
        )[0]
        self.training_df_old, self.test_df_old = get_encoded_logs(self.job)

    def test_radomise_features_single_feature(self):
        explanation_target = [[['prefix_2', 1]]]
        train_df, test_df = randomise_features(self.training_df_old.copy(), self.test_df_old.copy(), explanation_target)
        self.assertFalse(train_df.equals(self.training_df_old))
        self.assertFalse(train_df['prefix_2'].equals(self.training_df_old['prefix_2']))
        self.assertTrue(train_df.drop(['prefix_2'], 1).equals(self.training_df_old.drop(['prefix_2'], 1)))
        self.assertFalse(test_df.equals(self.test_df_old))
        # self.assertTrue(test_df['prefix_2'].equals(self.test_df_old['prefix_2']))
        self.assertTrue(test_df.drop(['prefix_2'], 1).equals(self.test_df_old.drop(['prefix_2'], 1)))

    def test_radomise_features_composed_feature(self):
        explanation_target = [[['prefix_2', 2], ['prefix_3', 1]]]
        train_df, test_df = randomise_features(self.training_df_old.copy(), self.test_df_old.copy(), explanation_target)
        self.assertFalse(train_df.equals(self.training_df_old))
        self.assertFalse(train_df['prefix_2'].equals(self.training_df_old['prefix_2']))
        self.assertFalse(train_df['prefix_3'].equals(self.training_df_old['prefix_3']))
        self.assertFalse(train_df.drop(['prefix_2'], 1).equals(self.training_df_old.drop(['prefix_2'], 1)))
        self.assertFalse(train_df.drop(['prefix_3'], 1).equals(self.training_df_old.drop(['prefix_3'], 1)))
        self.assertFalse(test_df.equals(self.test_df_old))
        # self.assertFalse(test_df['prefix_2'].equals(self.test_df_old['prefix_2']))
        # self.assertTrue(test_df['prefix_3'].equals(self.test_df_old['prefix_3']))
        self.assertFalse(test_df.drop(['prefix_2'], 1).equals(self.test_df_old.drop(['prefix_2'], 1)))
        self.assertFalse(test_df.drop(['prefix_3'], 1).equals(self.test_df_old.drop(['prefix_3'], 1)))

    def test_radomise_features_multiple_feature(self):
        explanation_target = [[['prefix_1', 1], ['prefix_2', 2]], [['prefix_3', 1]]]
        train_df, test_df = randomise_features(self.training_df_old.copy(), self.test_df_old.copy(), explanation_target)
        self.assertFalse(train_df.equals(self.training_df_old))
        self.assertTrue(train_df['prefix_1'].equals(self.training_df_old['prefix_1']))
        self.assertFalse(train_df['prefix_2'].equals(self.training_df_old['prefix_2']))
        self.assertFalse(train_df['prefix_3'].equals(self.training_df_old['prefix_3']))
        self.assertFalse(train_df.drop(['prefix_1'], 1).equals(self.training_df_old.drop(['prefix_1'], 1)))
        self.assertFalse(train_df.drop(['prefix_2'], 1).equals(self.training_df_old.drop(['prefix_2'], 1)))
        self.assertFalse(train_df.drop(['prefix_3'], 1).equals(self.training_df_old.drop(['prefix_3'], 1)))
        self.assertFalse(test_df.equals(self.test_df_old))
        self.assertTrue(test_df['prefix_1'].equals(self.test_df_old['prefix_1']))
        # self.assertFalse(test_df['prefix_2'].equals(self.test_df_old['prefix_2']))
        # self.assertFalse(test_df['prefix_3'].equals(self.test_df_old['prefix_3']))
        self.assertFalse(test_df.drop(['prefix_1'], 1).equals(self.test_df_old.drop(['prefix_1'], 1)))
        self.assertFalse(test_df.drop(['prefix_2'], 1).equals(self.test_df_old.drop(['prefix_2'], 1)))
        self.assertFalse(test_df.drop(['prefix_3'], 1).equals(self.test_df_old.drop(['prefix_3'], 1)))

    def test_save_randomised_set(self):
        initial_split_obj = self.job.split
        new_split = save_randomised_set(initial_split_obj)
        self.assertNotEqual(initial_split_obj.train_log.name, new_split.train_log.name)
        self.assertNotEqual(initial_split_obj.test_log.name, new_split.test_log.name)
        # TODO

    def test_explain(self):
        initial_result = {
            'f1_score': 0.7777777777777777,
            'accuracy': 0.75,
            'precision': 0.8333333333333334,
            'recall': 0.8333333333333334
        }
        explanation_target = [[['prefix_1', 1], ['prefix_2', 1], ['prefix_3', 1], ['prefix_4', 1]]]
        explanation = explain(self.exp, self.training_df_old, self.test_df_old, explanation_target)
        self.assertTrue(any([initial_result[key] != explanation['Retrain result'][key] for key in initial_result]))
