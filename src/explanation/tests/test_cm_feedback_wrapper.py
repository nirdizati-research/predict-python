
from django.test import TestCase
from pandas import DataFrame

from src.encoding.common import get_encoded_logs
from src.encoding.models import ValueEncodings
from src.explanation.cm_feedback_wrapper import explain, compute_confusion_matrix, retrieve_temporal_stability, \
    retrieve_lime_ts, process_explanations_in_feature_value_importance, filter_feature_value_importance, compute_data, mine_patterns, tassellate_numbers
from src.explanation.models import Explanation, ExplanationTypes
from src.jobs.models import JobTypes
from src.jobs.tasks import prediction_task
from src.labelling.models import LabelTypes
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.split.models import SplitTypes, SplitOrderingMethods
from src.utils.tests_utils import create_test_split, create_test_log, create_test_predictive_model, create_test_job, \
    create_test_encoding, create_test_labelling


class TestCmFeedbackWrapper(TestCase):
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
            labelling=create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value, attribute_name='label'),
            clustering=None,
            create_models=True,
            predictive_model=predictive_model,
            job_type=JobTypes.PREDICTION.value,
            hyperparameter_optimizer=None,
            incremental_train=None
        )

        prediction_task(self.job.id, do_publish_result=False)
        self.job.refresh_from_db()

        self.exp = Explanation.objects.get_or_create(
            type=ExplanationTypes.CMFEEDBACK.value,
            split=split,
            predictive_model=predictive_model,
            job=self.job,
            results={}
        )[0]
        self.training_df_old, self.test_df_old = get_encoded_logs(self.job)

    def test_tassellate_numbers_positive(self):
        element = '81.0'
        tassellated_element = tassellate_numbers(element)
        expected = '80'
        self.assertEqual(tassellated_element, expected)

    def test_tassellate_numbers_unnecessary1(self):
        element = 'Accept Claim'
        tassellated_element = tassellate_numbers(element)
        expected = 'Accept Claim'
        self.assertEqual(tassellated_element, expected)

    def test_tassellate_numbers_unnecessary2(self):
        element = 'asdf81.0'
        tassellated_element = tassellate_numbers(element)
        expected = 'asdf81.0'
        self.assertEqual(tassellated_element, expected)

    def test_retrieve_temporal_stability(self):
        ts = retrieve_temporal_stability(self.training_df_old, self.test_df_old.copy(), self.job, self.job.split)
        expected = {
            '2_122': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'High Insurance Check', 'predicted': 'false'}, 'prefix_3': {'value': 'High Medical History', 'predicted': 'false'}, 'prefix_4': {'value': 'Contact Hospital', 'predicted': 'false'}},
            '2_106': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Contact Hospital', 'predicted': 'false'}, 'prefix_3': {'value': 'High Insurance Check', 'predicted': 'false'}, 'prefix_4': {'value': 'Create Questionnaire', 'predicted': 'false'}},
            '2_107': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'true'}, 'prefix_3': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_4': {'value': 'Low Insurance Check', 'predicted': 'true'}},
            '2_108': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_3': {'value': 'Low Insurance Check', 'predicted': 'true'}, 'prefix_4': {'value': 'Accept Claim', 'predicted': 'true'}},
            '2_126': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'true'}, 'prefix_3': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_4': {'value': 'Low Insurance Check', 'predicted': 'true'}},
            '2_100': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'false'}, 'prefix_3': {'value': 'High Insurance Check', 'predicted': 'false'}, 'prefix_4': {'value': 'Contact Hospital', 'predicted': 'false'}},
            '2_124': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'false'}, 'prefix_3': {'value': 'High Medical History', 'predicted': 'false'}, 'prefix_4': {'value': 'Contact Hospital', 'predicted': 'false'}},
            '2_123': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'High Insurance Check', 'predicted': 'false'}, 'prefix_3': {'value': 'High Medical History', 'predicted': 'false'}, 'prefix_4': {'value': 'Contact Hospital', 'predicted': 'false'}},
            '2_103': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_3': {'value': 'Low Insurance Check', 'predicted': 'true'}, 'prefix_4': {'value': 'Accept Claim', 'predicted': 'true'}},
            '2_102': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'false'}, 'prefix_3': {'value': 'High Insurance Check', 'predicted': 'false'}, 'prefix_4': {'value': 'High Medical History', 'predicted': 'false'}},
            '2_104': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_3': {'value': 'Low Insurance Check', 'predicted': 'true'}, 'prefix_4': {'value': 'Accept Claim', 'predicted': 'true'}},
            '2_109': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'false'}, 'prefix_3': {'value': 'High Medical History', 'predicted': 'false'}, 'prefix_4': {'value': 'High Insurance Check', 'predicted': 'false'}},
            '2_101': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_3': {'value': 'Create Questionnaire', 'predicted': 'true'}, 'prefix_4': {'value': 'Low Insurance Check', 'predicted': 'true'}},
            '2_105': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Contact Hospital', 'predicted': 'false'}, 'prefix_3': {'value': 'High Medical History', 'predicted': 'false'}, 'prefix_4': {'value': 'High Insurance Check', 'predicted': 'false'}}
        }
        self.assertDictEqual(expected, ts)

    def test_compute_confusion_matrix(self):
        ts = {
            '2_100': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'false'}, 'prefix_3': {'value': 'High Insurance Check', 'predicted': 'false'}, 'prefix_4': {'value': 'Contact Hospital', 'predicted': 'false'}},
            '2_101': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_3': {'value': 'Create Questionnaire', 'predicted': 'true'}, 'prefix_4': {'value': 'Low Insurance Check', 'predicted': 'true'}},
            '2_102': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'false'}, 'prefix_3': {'value': 'High Insurance Check', 'predicted': 'false'}, 'prefix_4': {'value': 'High Medical History', 'predicted': 'false'}},
            '2_103': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_3': {'value': 'Low Insurance Check', 'predicted': 'true'}, 'prefix_4': {'value': 'Accept Claim', 'predicted': 'true'}},
            '2_104': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_3': {'value': 'Low Insurance Check', 'predicted': 'true'}, 'prefix_4': {'value': 'Accept Claim', 'predicted': 'true'}},
            '2_105': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Contact Hospital', 'predicted': 'false'}, 'prefix_3': {'value': 'High Medical History', 'predicted': 'false'}, 'prefix_4': {'value': 'High Insurance Check', 'predicted': 'false'}},
            '2_106': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Contact Hospital', 'predicted': 'false'}, 'prefix_3': {'value': 'High Insurance Check', 'predicted': 'false'}, 'prefix_4': {'value': 'Create Questionnaire', 'predicted': 'false'}},
            '2_107': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'true'}, 'prefix_3': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_4': {'value': 'Low Insurance Check', 'predicted': 'true'}},
            '2_108': {'prefix_1': {'value': 'Register', 'predicted': 'true'}, 'prefix_2': {'value': 'Low Medical History', 'predicted': 'true'}, 'prefix_3': {'value': 'Low Insurance Check', 'predicted': 'true'}, 'prefix_4': {'value': 'Accept Claim', 'predicted': 'true'}},
            '2_109': {'prefix_1': {'value': 'Register', 'predicted': 'false'}, 'prefix_2': {'value': 'Create Questionnaire', 'predicted': 'false'}, 'prefix_3': {'value': 'High Medical History', 'predicted': 'false'}, 'prefix_4': {'value': 'High Insurance Check', 'predicted': 'false'}}}
        gold = DataFrame(data=[
            {'trace_id': '2_100', 'label': 1},
            {'trace_id': '2_101', 'label': 2},
            {'trace_id': '2_102', 'label': 1},
            {'trace_id': '2_103', 'label': 2},
            {'trace_id': '2_104', 'label': 2},
            {'trace_id': '2_105', 'label': 1},
            {'trace_id': '2_106', 'label': 1},
            {'trace_id': '2_107', 'label': 1},   # <-- FP
            {'trace_id': '2_108', 'label': 2},
            {'trace_id': '2_109', 'label': 2}])  # <-- FN
        confusion_matrix = compute_confusion_matrix(ts, gold=gold, job_obj=self.job)
        expected = {
            'tp': ['2_103', '2_104', '2_108', '2_101'],
            'tn': ['2_105', '2_106', '2_100', '2_102'],
            'fp': ['2_107'],
            'fn': ['2_109']
        }
        self.assertTrue(sorted(expected['tp']) == sorted(confusion_matrix['tp']))
        self.assertTrue(sorted(expected['tn']) == sorted(confusion_matrix['tn']))
        self.assertTrue(sorted(expected['fp']) == sorted(confusion_matrix['fp']))
        self.assertTrue(sorted(expected['fn']) == sorted(confusion_matrix['fn']))

    def test_retrieve_lime_ts(self):
        lime_ts = retrieve_lime_ts(self.training_df_old, self.test_df_old.copy(), self.job, self.job.split)
        expected = {
            '2_122': {'prefix_4': {'prefix_3': {'value': 'High Medical History', 'importance': -0.21353833786944285}, 'prefix_2': {'value': 'High Insurance Check', 'importance': -0.1634805282078596}, 'prefix_4': {'value': 'Contact Hospital', 'importance': 0.004075908067748649}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_106': {'prefix_4': {'prefix_3': {'value': 'High Insurance Check', 'importance': -0.21802762064961867}, 'prefix_2': {'value': 'Contact Hospital', 'importance': -0.14026847974055287}, 'prefix_4': {'value': 'Create Questionnaire', 'importance': -0.0052520742546136555}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_107': {'prefix_4': {'prefix_3': {'value': 'Low Medical History', 'importance': 0.2823543161281943}, 'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.21258502082457842}, 'prefix_4': {'value': 'Low Insurance Check', 'importance': 0.0018408312810497063}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_108': {'prefix_4': {'prefix_2': {'value': 'Low Medical History', 'importance': 0.3080573983148055}, 'prefix_3': {'value': 'Low Insurance Check', 'importance': 0.29177589308712076}, 'prefix_4': {'value': 'Accept Claim', 'importance': 0.016406021937765063}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_126': {'prefix_4': {'prefix_3': {'value': 'Low Medical History', 'importance': 0.27824064724829844}, 'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.20340518385895157}, 'prefix_4': {'value': 'Low Insurance Check', 'importance': 0.006386185491340756}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_100': {'prefix_4': {'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.22891212261917707}, 'prefix_3': {'value': 'High Insurance Check', 'importance': -0.20249316188080696}, 'prefix_4': {'value': 'Contact Hospital', 'importance': 0.007645354094523817}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_124': {'prefix_4': {'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.218876310918173}, 'prefix_3': {'value': 'High Medical History', 'importance': -0.19064571883670475}, 'prefix_4': {'value': 'Contact Hospital', 'importance': 0.04383630913758562}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_123': {'prefix_4': {'prefix_3': {'value': 'High Medical History', 'importance': -0.2052171385847169}, 'prefix_2': {'value': 'High Insurance Check', 'importance': -0.14260102596172905}, 'prefix_4': {'value': 'Contact Hospital', 'importance': 0.021004293998412076}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_103': {'prefix_4': {'prefix_2': {'value': 'Low Medical History', 'importance': 0.2965207785352455}, 'prefix_3': {'value': 'Low Insurance Check', 'importance': 0.2709519560736247}, 'prefix_4': {'value': 'Accept Claim', 'importance': 0.027228937920124697}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_102': {'prefix_4': {'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.22252611869333622}, 'prefix_3': {'value': 'High Insurance Check', 'importance': -0.21627849641689711}, 'prefix_4': {'value': 'High Medical History', 'importance': 0.011617494080132082}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_104': {'prefix_4': {'prefix_2': {'value': 'Low Medical History', 'importance': 0.300586424314097}, 'prefix_3': {'value': 'Low Insurance Check', 'importance': 0.25728239044941903}, 'prefix_4': {'value': 'Accept Claim', 'importance': 0.01269068438901065}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_109': {'prefix_4': {'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.2134932761050843}, 'prefix_3': {'value': 'High Medical History', 'importance': -0.18709955714506707}, 'prefix_4': {'value': 'High Insurance Check', 'importance': 0.013420036251066655}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_101': {'prefix_4': {'prefix_2': {'value': 'Low Medical History', 'importance': 0.3292769105374355}, 'prefix_3': {'value': 'Create Questionnaire', 'importance': -0.22873401693027287}, 'prefix_4': {'value': 'Low Insurance Check', 'importance': 0.002956012816609118}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_105': {'prefix_4': {'prefix_3': {'value': 'High Medical History', 'importance': -0.19383890694348266}, 'prefix_2': {'value': 'Contact Hospital', 'importance': -0.12336107744753227}, 'prefix_4': {'value': 'High Insurance Check', 'importance': -0.0029920686039502527}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}}
        }
        #check same important features
        self.assertDictEqual(
            {
                tid: {
                    where: {
                        what: {
                            value : lime_ts[tid][where][what][value]
                            for value in lime_ts[tid][where][what] if value != 'importance'
                        }
                        for what in lime_ts[tid][where]
                    }
                    for where in lime_ts[tid]
                }
                for tid in lime_ts
            },
            {
                tid: {
                    where: {
                        what: {
                            value : expected[tid][where][what][value]
                            for value in expected[tid][where][what] if value != 'importance'
                        }
                        for what in expected[tid][where]
                    }
                    for where in expected[tid]
                }
                for tid in expected
            }
        )
        # check same polarity of important features
        self.assertDictEqual(
            {
                tid: {
                    where: {
                        what: {
                            value : lime_ts[tid][where][what][value] if value != 'importance'
                            else 'positive' if str(lime_ts[tid][where][what][value])[0] != '-'
                                else 'negative'
                            for value in lime_ts[tid][where][what]
                        }
                        for what in lime_ts[tid][where]
                    }
                    for where in lime_ts[tid]
                }
                for tid in lime_ts
            },
            {
                tid: {
                    where: {
                        what: {
                            value : expected[tid][where][what][value] if value != 'importance'
                            else 'positive' if str(lime_ts[tid][where][what][value])[0] != '-'
                                else 'negative'
                            for value in expected[tid][where][what]
                        }
                        for what in expected[tid][where]
                    }
                    for where in expected[tid]
                }
                for tid in expected
            }
        )

    def test_process_lime_features(self):
        self.confusion_matrix = {
            'tp': ['2_103', '2_104', '2_108', '2_101'],
            'tn': ['2_105', '2_106', '2_100', '2_102', '2_123'],
            'fp': ['2_107', '2_126'],
            'fn': ['2_109', '2_122', '2_124']
        }
        self.lime_ts = {
            '2_122': {'prefix_4': {'prefix_3': {'value': 'High Medical History', 'importance': -0.21353833786944285}, 'prefix_2': {'value': 'High Insurance Check', 'importance': -0.1634805282078596}, 'prefix_4': {'value': 'Contact Hospital', 'importance': 0.004075908067748649}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_106': {'prefix_4': {'prefix_3': {'value': 'High Insurance Check', 'importance': -0.21802762064961867}, 'prefix_2': {'value': 'Contact Hospital', 'importance': -0.14026847974055287}, 'prefix_4': {'value': 'Create Questionnaire', 'importance': -0.0052520742546136555}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_107': {'prefix_4': {'prefix_3': {'value': 'Low Medical History', 'importance': 0.2823543161281943}, 'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.21258502082457842}, 'prefix_4': {'value': 'Low Insurance Check', 'importance': 0.0018408312810497063}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_108': {'prefix_4': {'prefix_2': {'value': 'Low Medical History', 'importance': 0.3080573983148055}, 'prefix_3': {'value': 'Low Insurance Check', 'importance': 0.29177589308712076}, 'prefix_4': {'value': 'Accept Claim', 'importance': 0.016406021937765063}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_126': {'prefix_4': {'prefix_3': {'value': 'Low Medical History', 'importance': 0.27824064724829844}, 'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.20340518385895157}, 'prefix_4': {'value': 'Low Insurance Check', 'importance': 0.006386185491340756}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_100': {'prefix_4': {'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.22891212261917707}, 'prefix_3': {'value': 'High Insurance Check', 'importance': -0.20249316188080696}, 'prefix_4': {'value': 'Contact Hospital', 'importance': 0.007645354094523817}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_124': {'prefix_4': {'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.218876310918173}, 'prefix_3': {'value': 'High Medical History', 'importance': -0.19064571883670475}, 'prefix_4': {'value': 'Contact Hospital', 'importance': 0.04383630913758562}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_123': {'prefix_4': {'prefix_3': {'value': 'High Medical History', 'importance': -0.2052171385847169}, 'prefix_2': {'value': 'High Insurance Check', 'importance': -0.14260102596172905}, 'prefix_4': {'value': 'Contact Hospital', 'importance': 0.021004293998412076}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_103': {'prefix_4': {'prefix_2': {'value': 'Low Medical History', 'importance': 0.2965207785352455}, 'prefix_3': {'value': 'Low Insurance Check', 'importance': 0.2709519560736247}, 'prefix_4': {'value': 'Accept Claim', 'importance': 0.027228937920124697}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_102': {'prefix_4': {'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.22252611869333622}, 'prefix_3': {'value': 'High Insurance Check', 'importance': -0.21627849641689711}, 'prefix_4': {'value': 'High Medical History', 'importance': 0.011617494080132082}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_104': {'prefix_4': {'prefix_2': {'value': 'Low Medical History', 'importance': 0.300586424314097}, 'prefix_3': {'value': 'Low Insurance Check', 'importance': 0.25728239044941903}, 'prefix_4': {'value': 'Accept Claim', 'importance': 0.01269068438901065}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_109': {'prefix_4': {'prefix_2': {'value': 'Create Questionnaire', 'importance': -0.2134932761050843}, 'prefix_3': {'value': 'High Medical History', 'importance': -0.18709955714506707}, 'prefix_4': {'value': 'High Insurance Check', 'importance': 0.013420036251066655}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_101': {'prefix_4': {'prefix_2': {'value': 'Low Medical History', 'importance': 0.3292769105374355}, 'prefix_3': {'value': 'Create Questionnaire', 'importance': -0.22873401693027287}, 'prefix_4': {'value': 'Low Insurance Check', 'importance': 0.002956012816609118}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}},
            '2_105': {'prefix_4': {'prefix_3': {'value': 'High Medical History', 'importance': -0.19383890694348266}, 'prefix_2': {'value': 'Contact Hospital', 'importance': -0.12336107744753227}, 'prefix_4': {'value': 'High Insurance Check', 'importance': -0.0029920686039502527}, 'prefix_1': {'value': 'Register', 'importance': 0.0}}}
        }
        limefeats = process_lime_features(self.lime_ts, self.confusion_matrix, ['tp', 'tn', 'fp', 'fn'], self.job.encoding.prefix_length)
        expected = {
            'tp': {
                '2_108': [('prefix_2', 'Low Medical History', 0.3080573983148055), ('prefix_3', 'Low Insurance Check', 0.29177589308712076), ('prefix_4', 'Accept Claim', 0.016406021937765063), ('prefix_1', 'Register', 0.0)],
                '2_103': [('prefix_2', 'Low Medical History', 0.2965207785352455), ('prefix_3', 'Low Insurance Check', 0.2709519560736247), ('prefix_4', 'Accept Claim', 0.027228937920124697), ('prefix_1', 'Register', 0.0)],
                '2_104': [('prefix_2', 'Low Medical History', 0.300586424314097), ('prefix_3', 'Low Insurance Check', 0.25728239044941903), ('prefix_4', 'Accept Claim', 0.01269068438901065), ('prefix_1', 'Register', 0.0)],
                '2_101': [('prefix_2', 'Low Medical History', 0.3292769105374355), ('prefix_4', 'Low Insurance Check', 0.002956012816609118), ('prefix_1', 'Register', 0.0), ('prefix_3', 'Create Questionnaire', -0.22873401693027287)]},
            'tn': {
                '2_106': [('prefix_3', 'High Insurance Check', -0.21802762064961867), ('prefix_2', 'Contact Hospital', -0.14026847974055287), ('prefix_4', 'Create Questionnaire', -0.0052520742546136555), ('prefix_1', 'Register', 0.0)],
                '2_100': [('prefix_2', 'Create Questionnaire', -0.22891212261917707), ('prefix_3', 'High Insurance Check', -0.20249316188080696), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.007645354094523817)],
                '2_123': [('prefix_3', 'High Medical History', -0.2052171385847169), ('prefix_2', 'High Insurance Check', -0.14260102596172905), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.021004293998412076)],
                '2_102': [('prefix_2', 'Create Questionnaire', -0.22252611869333622), ('prefix_3', 'High Insurance Check', -0.21627849641689711), ('prefix_1', 'Register', 0.0), ('prefix_4', 'High Medical History', 0.011617494080132082)],
                '2_105': [('prefix_3', 'High Medical History', -0.19383890694348266), ('prefix_2', 'Contact Hospital', -0.12336107744753227), ('prefix_4', 'High Insurance Check', -0.0029920686039502527), ('prefix_1', 'Register', 0.0)]},
            'fp': {
                '2_107': [('prefix_3', 'Low Medical History', 0.2823543161281943), ('prefix_4', 'Low Insurance Check', 0.0018408312810497063), ('prefix_1', 'Register', 0.0), ('prefix_2', 'Create Questionnaire', -0.21258502082457842)],
                '2_126': [('prefix_3', 'Low Medical History', 0.27824064724829844), ('prefix_4', 'Low Insurance Check', 0.006386185491340756), ('prefix_1', 'Register', 0.0), ('prefix_2', 'Create Questionnaire', -0.20340518385895157)]},
            'fn': {
                '2_122': [('prefix_3', 'High Medical History', -0.21353833786944285), ('prefix_2', 'High Insurance Check', -0.1634805282078596), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.004075908067748649)],
                '2_124': [('prefix_2', 'Create Questionnaire', -0.218876310918173), ('prefix_3', 'High Medical History', -0.19064571883670475), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.04383630913758562)],
                '2_109': [('prefix_2', 'Create Questionnaire', -0.2134932761050843), ('prefix_3', 'High Medical History', -0.18709955714506707), ('prefix_1', 'Register', 0.0), ('prefix_4', 'High Insurance Check', 0.013420036251066655)]}
        }
        self.assertDictEqual(
            {
                tid: sorted(expected['tp'][tid])
                for tid in expected['tp']
            },
            {
                tid: sorted(limefeats['tp'][tid])
                for tid in limefeats['tp']
            }
        )
        self.assertDictEqual(
            {
                tid: sorted(expected['tn'][tid])
                for tid in expected['tn']
            },
            {
                tid: sorted(limefeats['tn'][tid])
                for tid in limefeats['tn']
            }
        )
        self.assertDictEqual(
            {
                tid: sorted(expected['fp'][tid])
                for tid in expected['fp']
            },
            {
                tid: sorted(limefeats['fp'][tid])
                for tid in limefeats['fp']
            }
        )
        self.assertDictEqual(
            {
                tid: sorted(expected['fn'][tid])
                for tid in expected['fn']
            },
            {
                tid: sorted(limefeats['fn'][tid])
                for tid in limefeats['fn']
            }
        ) #todo just test if they are positive or negative?

    def test_filter_lime_features(self):
        self.limefeats = {
            'tp': {
                '2_108': [('prefix_2', 'Low Medical History', 0.3080573983148055), ('prefix_3', 'Low Insurance Check', 0.29177589308712076), ('prefix_4', 'Accept Claim', 0.016406021937765063), ('prefix_1', 'Register', 0.0)],
                '2_103': [('prefix_2', 'Low Medical History', 0.2965207785352455), ('prefix_3', 'Low Insurance Check', 0.2709519560736247), ('prefix_4', 'Accept Claim', 0.027228937920124697), ('prefix_1', 'Register', 0.0)],
                '2_104': [('prefix_2', 'Low Medical History', 0.300586424314097), ('prefix_3', 'Low Insurance Check', 0.25728239044941903), ('prefix_4', 'Accept Claim', 0.01269068438901065), ('prefix_1', 'Register', 0.0)],
                '2_101': [('prefix_2', 'Low Medical History', 0.3292769105374355), ('prefix_4', 'Low Insurance Check', 0.002956012816609118), ('prefix_1', 'Register', 0.0), ('prefix_3', 'Create Questionnaire', -0.22873401693027287)]},
            'tn': {
                '2_106': [('prefix_3', 'High Insurance Check', -0.21802762064961867), ('prefix_2', 'Contact Hospital', -0.14026847974055287), ('prefix_4', 'Create Questionnaire', -0.0052520742546136555), ('prefix_1', 'Register', 0.0)],
                '2_100': [('prefix_2', 'Create Questionnaire', -0.22891212261917707), ('prefix_3', 'High Insurance Check', -0.20249316188080696), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.007645354094523817)],
                '2_123': [('prefix_3', 'High Medical History', -0.2052171385847169), ('prefix_2', 'High Insurance Check', -0.14260102596172905), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.021004293998412076)],
                '2_102': [('prefix_2', 'Create Questionnaire', -0.22252611869333622), ('prefix_3', 'High Insurance Check', -0.21627849641689711), ('prefix_1', 'Register', 0.0), ('prefix_4', 'High Medical History', 0.011617494080132082)],
                '2_105': [('prefix_3', 'High Medical History', -0.19383890694348266), ('prefix_2', 'Contact Hospital', -0.12336107744753227), ('prefix_4', 'High Insurance Check', -0.0029920686039502527), ('prefix_1', 'Register', 0.0)]},
            'fp': {
                '2_107': [('prefix_3', 'Low Medical History', 0.2823543161281943), ('prefix_4', 'Low Insurance Check', 0.0018408312810497063), ('prefix_1', 'Register', 0.0), ('prefix_2', 'Create Questionnaire', -0.21258502082457842)],
                '2_126': [('prefix_3', 'Low Medical History', 0.27824064724829844), ('prefix_4', 'Low Insurance Check', 0.006386185491340756), ('prefix_1', 'Register', 0.0), ('prefix_2', 'Create Questionnaire', -0.20340518385895157)]},
            'fn': {
                '2_122': [('prefix_3', 'High Medical History', -0.21353833786944285), ('prefix_2', 'High Insurance Check', -0.1634805282078596), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.004075908067748649)],
                '2_124': [('prefix_2', 'Create Questionnaire', -0.218876310918173), ('prefix_3', 'High Medical History', -0.19064571883670475), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.04383630913758562)],
                '2_109': [('prefix_2', 'Create Questionnaire', -0.2134932761050843), ('prefix_3', 'High Medical History', -0.18709955714506707), ('prefix_1', 'Register', 0.0), ('prefix_4', 'High Insurance Check', 0.013420036251066655)]}
        }
        filtered_limefeats = filter_lime_features(
            self.limefeats,
            LIMEFEATS={
                'abs_lime': False,
                'tp': 0.2,
                'tn': 0.2,
                'fp': 0.2,
                'fn': 0.2
            },
            CONFUSION_MATRIX=['tp', 'tn', 'fp', 'fn']
        )
        expected = {
            'tp': {'2_108': [('prefix_2', 'Low Medical History', 0.3080573983148055), ('prefix_3', 'Low Insurance Check', 0.29177589308712076)], '2_103': [('prefix_2', 'Low Medical History', 0.2965207785352455), ('prefix_3', 'Low Insurance Check', 0.2709519560736247)], '2_104': [('prefix_2', 'Low Medical History', 0.300586424314097), ('prefix_3', 'Low Insurance Check', 0.25728239044941903)], '2_101': [('prefix_2', 'Low Medical History', 0.3292769105374355)]},
            'tn': {'2_106': [('prefix_3', 'High Insurance Check', -0.21802762064961867)], '2_100': [('prefix_2', 'Create Questionnaire', -0.22891212261917707), ('prefix_3', 'High Insurance Check', -0.20249316188080696)], '2_123': [('prefix_3', 'High Medical History', -0.2052171385847169)], '2_102': [('prefix_2', 'Create Questionnaire', -0.22252611869333622), ('prefix_3', 'High Insurance Check', -0.21627849641689711)]},
            'fp': {'2_107': [('prefix_3', 'Low Medical History', 0.2823543161281943)], '2_126': [('prefix_3', 'Low Medical History', 0.27824064724829844)]},
            'fn': {'2_122': [('prefix_3', 'High Medical History', -0.21353833786944285)], '2_124': [('prefix_2', 'Create Questionnaire', -0.218876310918173)], '2_109': [('prefix_2', 'Create Questionnaire', -0.2134932761050843)]}
        }
        self.assertDictEqual(expected, filtered_limefeats) # todo just test if they are greater than the threshold

    def test_compute_data(self):
        self.limefeats = {
            'tp': {
                '2_108': [('prefix_2', 'Low Medical History', 0.3080573983148055), ('prefix_3', 'Low Insurance Check', 0.29177589308712076), ('prefix_4', 'Accept Claim', 0.016406021937765063), ('prefix_1', 'Register', 0.0)],
                '2_103': [('prefix_2', 'Low Medical History', 0.2965207785352455), ('prefix_3', 'Low Insurance Check', 0.2709519560736247), ('prefix_4', 'Accept Claim', 0.027228937920124697), ('prefix_1', 'Register', 0.0)],
                '2_104': [('prefix_2', 'Low Medical History', 0.300586424314097), ('prefix_3', 'Low Insurance Check', 0.25728239044941903), ('prefix_4', 'Accept Claim', 0.01269068438901065), ('prefix_1', 'Register', 0.0)],
                '2_101': [('prefix_2', 'Low Medical History', 0.3292769105374355), ('prefix_4', 'Low Insurance Check', 0.002956012816609118), ('prefix_1', 'Register', 0.0), ('prefix_3', 'Create Questionnaire', -0.22873401693027287)]},
            'tn': {
                '2_106': [('prefix_3', 'High Insurance Check', -0.21802762064961867), ('prefix_2', 'Contact Hospital', -0.14026847974055287), ('prefix_4', 'Create Questionnaire', -0.0052520742546136555), ('prefix_1', 'Register', 0.0)],
                '2_100': [('prefix_2', 'Create Questionnaire', -0.22891212261917707), ('prefix_3', 'High Insurance Check', -0.20249316188080696), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.007645354094523817)],
                '2_123': [('prefix_3', 'High Medical History', -0.2052171385847169), ('prefix_2', 'High Insurance Check', -0.14260102596172905), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.021004293998412076)],
                '2_102': [('prefix_2', 'Create Questionnaire', -0.22252611869333622), ('prefix_3', 'High Insurance Check', -0.21627849641689711), ('prefix_1', 'Register', 0.0), ('prefix_4', 'High Medical History', 0.011617494080132082)],
                '2_105': [('prefix_3', 'High Medical History', -0.19383890694348266), ('prefix_2', 'Contact Hospital', -0.12336107744753227), ('prefix_4', 'High Insurance Check', -0.0029920686039502527), ('prefix_1', 'Register', 0.0)]},
            'fp': {
                '2_107': [('prefix_3', 'Low Medical History', 0.2823543161281943), ('prefix_4', 'Low Insurance Check', 0.0018408312810497063), ('prefix_1', 'Register', 0.0), ('prefix_2', 'Create Questionnaire', -0.21258502082457842)],
                '2_126': [('prefix_3', 'Low Medical History', 0.27824064724829844), ('prefix_4', 'Low Insurance Check', 0.006386185491340756), ('prefix_1', 'Register', 0.0), ('prefix_2', 'Create Questionnaire', -0.20340518385895157)]},
            'fn': {
                '2_122': [('prefix_3', 'High Medical History', -0.21353833786944285), ('prefix_2', 'High Insurance Check', -0.1634805282078596), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.004075908067748649)],
                '2_124': [('prefix_2', 'Create Questionnaire', -0.218876310918173), ('prefix_3', 'High Medical History', -0.19064571883670475), ('prefix_1', 'Register', 0.0), ('prefix_4', 'Contact Hospital', 0.04383630913758562)],
                '2_109': [('prefix_2', 'Create Questionnaire', -0.2134932761050843), ('prefix_3', 'High Medical History', -0.18709955714506707), ('prefix_1', 'Register', 0.0), ('prefix_4', 'High Insurance Check', 0.013420036251066655)]}
        }
        self.filtered_limefeats = {
            'tp': {'2_108': [('prefix_2', 'Low Medical History', 0.3080573983148055), ('prefix_3', 'Low Insurance Check', 0.29177589308712076)], '2_103': [('prefix_2', 'Low Medical History', 0.2965207785352455), ('prefix_3', 'Low Insurance Check', 0.2709519560736247)], '2_104': [('prefix_2', 'Low Medical History', 0.300586424314097), ('prefix_3', 'Low Insurance Check', 0.25728239044941903)], '2_101': [('prefix_2', 'Low Medical History', 0.3292769105374355)]},
            'tn': {'2_106': [('prefix_3', 'High Insurance Check', -0.21802762064961867)], '2_100': [('prefix_2', 'Create Questionnaire', -0.22891212261917707), ('prefix_3', 'High Insurance Check', -0.20249316188080696)], '2_123': [('prefix_3', 'High Medical History', -0.2052171385847169)], '2_102': [('prefix_2', 'Create Questionnaire', -0.22252611869333622), ('prefix_3', 'High Insurance Check', -0.21627849641689711)]},
            'fp': {'2_107': [('prefix_3', 'Low Medical History', 0.2823543161281943)], '2_126': [('prefix_3', 'Low Medical History', 0.27824064724829844)]},
            'fn': {'2_122': [('prefix_3', 'High Medical History', -0.21353833786944285)], '2_124': [('prefix_2', 'Create Questionnaire', -0.218876310918173)], '2_109': [('prefix_2', 'Create Questionnaire', -0.2134932761050843)]}
        }
        data = compute_data(['tp', 'tn', 'fp', 'fn'], self.limefeats, self.filtered_limefeats)
        expected = {
            'tp': {'2_108': ['prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'], '2_103': ['prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'], '2_104': ['prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'], '2_101': ['prefix_2_Low Medical History']},
            'tn': {'2_106': ['prefix_3_High Insurance Check'], '2_100': ['prefix_2_Create Questionnaire', 'prefix_3_High Insurance Check'], '2_123': ['prefix_3_High Medical History'], '2_102': ['prefix_2_Create Questionnaire', 'prefix_3_High Insurance Check'], '2_105': []},
            'fp': {'2_107': ['prefix_3_Low Medical History'], '2_126': ['prefix_3_Low Medical History']},
            'fn': {'2_122': ['prefix_3_High Medical History'], '2_124': ['prefix_2_Create Questionnaire'], '2_109': ['prefix_2_Create Questionnaire']}}
        self.assertDictEqual(expected, data)

    def test_mine_patterns(self):
        data = {
            'tp': {'2_108': ['prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'], '2_103': ['prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'], '2_104': ['prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'], '2_101': ['prefix_2_Low Medical History']},
            'tn': {'2_106': ['prefix_3_High Insurance Check'], '2_100': ['prefix_2_Create Questionnaire', 'prefix_3_High Insurance Check'], '2_123': ['prefix_3_High Medical History'], '2_102': ['prefix_2_Create Questionnaire', 'prefix_3_High Insurance Check'], '2_105': []},
            'fp': {'2_107': ['prefix_3_Low Medical History'], '2_126': ['prefix_3_Low Medical History']},
            'fn': {'2_122': ['prefix_3_High Medical History'], '2_124': ['prefix_2_Create Questionnaire'], '2_109': ['prefix_2_Create Questionnaire']}}
        frequent_patterns = mine_patterns(data, MINING_METHOD='item_mining', CONFUSION_MATRIX=['tp', 'tn', 'fp', 'fn'])
        expected = {
            'tp': [(('prefix_3_Low Insurance Check',), 3), (('prefix_3_Low Insurance Check', 'prefix_2_Low Medical History'), 3), (('prefix_2_Low Medical History',), 4)],
            'tn': [(('prefix_2_Create Questionnaire',), 2), (('prefix_3_High Insurance Check', 'prefix_2_Create Questionnaire'), 2), (('prefix_3_High Insurance Check',), 3)],
            'fp': [(('prefix_3_Low Medical History',), 2)],
            'fn': [(('prefix_2_Create Questionnaire',), 2)]}
        self.assertEqual(expected['fp'], frequent_patterns['fp'])
        self.assertEqual(expected['fn'], frequent_patterns['fn'])
        self.assertEqual([set(element) for element, score in expected['tp']], [set(element) for element, score in frequent_patterns['tp']])
        self.assertEqual([set(element) for element, score in expected['tn']], [set(element) for element, score in frequent_patterns['tn']])
        self.assertEqual([score for element, score in expected['tn']], [score for element, score in frequent_patterns['tn']])

    # def test_retrieve_right_len(self):
    #     self.assertFalse(True)
    #
    # def test_weight_freq_seqs(self):
    #     self.assertFalse(True)

    def test_explain(self):
        mined_patterns = explain(self.exp, self.training_df_old, self.test_df_old, top_k=3, prefix_target=None)
        expected = {
            'confusion_matrix': {
                'tp': ['2_108', '2_103', '2_101', '2_104'],
                'tn': ['2_102', '2_105', '2_106', '2_123', '2_100'],
                'fp': ['2_107', '2_126'],
                'fn': ['2_124', '2_122', '2_109']
            },
            'data': {
                'tp': {'2_108': ['prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'], '2_103': ['prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'], '2_101': ['prefix_2_Low Medical History'], '2_104': ['prefix_2_Low Medical History', 'prefix_3_Low Insurance Check']},
                'tn': {'2_102': [], '2_105': ['prefix_3_High Medical History'], '2_106': [], '2_123': ['prefix_3_High Medical History'], '2_100': ['prefix_3_High Insurance Check']},
                'fp': {'2_107': ['prefix_3_Low Medical History'], '2_126': ['prefix_3_Low Medical History']},
                'fn': {'2_124': ['prefix_3_High Medical History'], '2_122': ['prefix_3_High Medical History'], '2_109': ['prefix_3_High Medical History']}
            },
            'freq_seqs_after_filter': {
                'tp': [(('prefix_3_Low Insurance Check',), 3), (('prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'), 3), (('prefix_2_Low Medical History',), 4)],
                'tn': [(('prefix_3_High Medical History',), 2)],
                'fp': [(('prefix_3_Low Medical History',), 2)],
                'fn': [(('prefix_3_High Medical History',), 3)]
            },
            'filtered_freq_seqs_after_filter': {
                'tp': [[('prefix_2_Low Medical History',), 1.0], [('prefix_3_Low Insurance Check',), 0.75], [('prefix_2_Low Medical History', 'prefix_3_Low Insurance Check'), 0.75]],
                'tn': [[('prefix_3_High Medical History',), 0.4]],
                'fp': [[('prefix_3_Low Medical History',), 1.0]],
                'fn': [[('prefix_3_High Medical History',), 1.0]]
            }
        }

        self.assertTrue(sorted(expected['confusion_matrix']) == sorted(mined_patterns['confusion_matrix']))
        self.assertTrue(sorted(expected['data']) == sorted(mined_patterns['data']))
        self.assertTrue(sorted(expected['freq_seqs_after_filter']) == sorted(mined_patterns['freq_seqs_after_filter']))
        self.assertTrue(sorted(expected['filtered_freq_seqs_after_filter']) == sorted(mined_patterns['filtered_freq_seqs_after_filter']))
