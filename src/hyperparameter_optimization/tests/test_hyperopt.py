"""
hyperopt tests
"""

from django.test import TestCase
from hyperopt import Trials, Ctrl, STATUS_OK
from pandas import DataFrame

from src.hyperparameter_optimization import hyperopt_wrapper
from src.hyperparameter_optimization.hyperopt_spaces import _get_space
from src.hyperparameter_optimization.hyperopt_wrapper import calculate_hyperopt, _retrieve_train_validate_test, \
    _run_hyperoptimisation, OPTIMISATION_ALGORITHM, _test_best_candidate
from src.hyperparameter_optimization.models import HyperOptLosses
from src.labelling.models import LabelTypes, ThresholdTypes
from src.predictive_model.classification.classification import _prepare_results as classification_prepare_results
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.predictive_model.regression.models import RegressionMethods
from src.utils.result_metrics import _prepare_results as regression_prepare_results
from src.utils.tests_utils import create_test_predictive_model, create_test_job, create_test_hyperparameter_optimizer, \
    create_test_encoding, create_test_labelling


class TestHyperopt(TestCase):
    """Proof of concept tests"""

    @staticmethod
    def get_classification_job(predictive_model: str, prediction_method: str, metric: HyperOptLosses = HyperOptLosses.ACC.value):
        encoding = create_test_encoding(prefix_length=8, padding=True)
        pred_model = create_test_predictive_model(predictive_model=predictive_model,
                                                  prediction_method=prediction_method)
        hyperparameter_optimizer = create_test_hyperparameter_optimizer(performance_metric=metric)

        job = create_test_job(predictive_model=pred_model,
                              encoding=encoding,
                              hyperparameter_optimizer=hyperparameter_optimizer)
        return job

    @staticmethod
    def get_regression_job(predictive_model: str, prediction_method: str,
                           metric: HyperOptLosses = HyperOptLosses.ACC.value):
        encoding = create_test_encoding(prefix_length=8, padding=True)
        pred_model = create_test_predictive_model(predictive_model=predictive_model,
                                                  prediction_method=prediction_method)
        hyperparameter_optimizer = create_test_hyperparameter_optimizer(performance_metric=metric)

        job = create_test_job(predictive_model=pred_model,
                              encoding=encoding,
                              labelling=create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value,
                                                              threshold_type=ThresholdTypes.NONE.value),
                              hyperparameter_optimizer=hyperparameter_optimizer)
        return job

    def test_class_randomForest(self):
        job = self.get_classification_job(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.RANDOM_FOREST.value)
        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    # def test_class_knn(self):
    #     job = self.get_job(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.KNN.value)
    #
    #     results, _ = calculate_hyperopt(job)
    #     self.assertIsNotNone(results)

    def test_class_xgboost(self):
        job = self.get_classification_job(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.XGBOOST.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    def test_class_decision_tree(self):
        job = self.get_classification_job(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.DECISION_TREE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    def test_regression_random_forest(self):
        job = self.get_regression_job(PredictiveModels.REGRESSION.value, RegressionMethods.RANDOM_FOREST.value,
                                          HyperOptLosses.RMSE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    def test_regression_linear(self):
        job = self.get_regression_job(PredictiveModels.REGRESSION.value, RegressionMethods.LINEAR.value,
                                          HyperOptLosses.RMSE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    def test_regression_lasso(self):
        job = self.get_regression_job(PredictiveModels.REGRESSION.value, RegressionMethods.LASSO.value,
                                          HyperOptLosses.RMSE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    def test_regression_xgboost(self):
        job = self.get_regression_job(PredictiveModels.REGRESSION.value, RegressionMethods.XGBOOST.value,
                                          HyperOptLosses.RMSE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)


class TestHyperoptFunctionsClassification(TestCase):
    """Proof of concept tests"""

    def setUp(self):
        self.job = self.get_classification_job(PredictiveModels.CLASSIFICATION.value,
                                               ClassificationMethods.PERCEPTRON.value)
        _,_,_ = calculate_hyperopt(self.job)

    @staticmethod
    def get_classification_job(predictive_model: str, prediction_method: str,
                               metric: HyperOptLosses = HyperOptLosses.ACC.value):
        encoding = create_test_encoding(prefix_length=7, padding=True)
        pred_model = create_test_predictive_model(predictive_model=predictive_model,
                                                  prediction_method=prediction_method)
        hyperparameter_optimizer = create_test_hyperparameter_optimizer(performance_metric=metric)

        job = create_test_job(predictive_model=pred_model,
                              encoding=encoding,
                              hyperparameter_optimizer=hyperparameter_optimizer)
        return job

    @staticmethod
    def create_trials(job):
        trials = Trials()
        space = _get_space(job)
        algorithm = OPTIMISATION_ALGORITHM[
            job.hyperparameter_optimizer.__getattribute__(
                job.hyperparameter_optimizer.optimization_method.lower()
            ).algorithm_type
        ]
        calculate_hyperopt.global_job = job
        max_evaluations = 5
        _run_hyperoptimisation(space, algorithm.suggest, max_evaluations, trials)
        return trials

    def test_retrieve_train_validate_test(self):
        local_train_df = DataFrame([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]],
                                   columns=['A', 'B'])
        local_test_df = DataFrame([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]],
                                   columns=['A', 'B'])
        train, validation, test = _retrieve_train_validate_test(local_train_df, local_test_df)

        self.assertDictEqual(DataFrame([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1],
                                    [1, 0]], columns=['A', 'B']).to_dict(), DataFrame(train).to_dict())
        self.assertDictEqual(DataFrame([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0],
                                        [0, 1], [1, 0]], columns=['A', 'B']).to_dict(), DataFrame(validation).to_dict())
        self.assertDictEqual(DataFrame([[0, 1], [1, 0]], index=[8, 9], columns=['A', 'B']).to_dict(), DataFrame(test).to_dict())

    def test_run_hyperoptimisation(self):
        trials = self.create_trials(self.job)

        self.assertGreater(len(trials), 0)
        self.assertEqual(len(trials), 5)

    def test_test_best_candidate_classification(self):
        trials = self.create_trials(self.job)

        results_df, auc = _test_best_candidate(trials.best_trial['result'], self.job.labelling.type,
                                               self.job.predictive_model.predictive_model)
        results = classification_prepare_results(results_df, auc)
        del trials.best_trial['result']['results']['elapsed_time']
        self.assertDictEqual(trials.best_trial['result']['results'], results)

    def test_best_candidate_classification(self):
        trials = self.create_trials(self.job)
        best_candidate = trials.best_trial['result']

        best_trial_loss = sorted(list(trials), key=lambda item: item['result']['loss'])[0]['result']
        best_trial_acc = sorted(list(trials), key=lambda item: item['result']['results']['acc'],
                                reverse=True)[0]['result']

        self.assertDictEqual(best_candidate, best_trial_loss)
        self.assertDictEqual(best_candidate, best_trial_acc)



class TestHyperoptFunctionsRegression(TestCase):
    """Proof of concept tests"""

    def setUp(self):
        self.job = self.get_regression_job(PredictiveModels.REGRESSION.value, RegressionMethods.XGBOOST.value)
        _,_,_ = calculate_hyperopt(self.job)

    @staticmethod
    def get_regression_job(predictive_model: str, prediction_method: str,
                           metric: HyperOptLosses = HyperOptLosses.MAE.value):
        encoding = create_test_encoding(prefix_length=8, padding=True)
        pred_model = create_test_predictive_model(predictive_model=predictive_model,
                                                  prediction_method=prediction_method)
        hyperparameter_optimizer = create_test_hyperparameter_optimizer(performance_metric=metric)

        job = create_test_job(predictive_model=pred_model,
                              encoding=encoding,
                              labelling=create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value,
                                                              threshold_type=ThresholdTypes.NONE.value),
                              hyperparameter_optimizer=hyperparameter_optimizer)
        return job

    @staticmethod
    def create_trials(job):
        trials = Trials()
        space = _get_space(job)
        algorithm = OPTIMISATION_ALGORITHM[
            job.hyperparameter_optimizer.__getattribute__(
                job.hyperparameter_optimizer.optimization_method.lower()
            ).algorithm_type
        ]
        calculate_hyperopt.global_job = job
        max_evaluations = 5
        _run_hyperoptimisation(space, algorithm.suggest, max_evaluations, trials)
        return trials

    def test_best_candidate_regression(self):
        trials = self.create_trials(self.job)
        best_candidate = trials.best_trial['result']

        best_trial_loss = sorted(list(trials), key=lambda item: item['result']['loss'])[0]['result']
        best_trial_mae = sorted(list(trials), key=lambda item: item['result']['results']['mae'],
                                reverse=True)[0]['result']

        self.assertDictEqual(best_candidate, best_trial_loss)
        self.assertDictEqual(best_candidate, best_trial_mae)

    def test_test_best_candidate_regression(self):
        trials = self.create_trials(self.job)

        results_df, auc = _test_best_candidate(trials.best_trial['result'], self.job.labelling.type,
                                               self.job.predictive_model.predictive_model)
        results = regression_prepare_results(results_df, self.job.labelling)
        del trials.best_trial['result']['results']['elapsed_time']
        self.assertDictEqual(trials.best_trial['result']['results'], results)
