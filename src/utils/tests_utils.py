from src.clustering.models import Clustering, ClusteringMethods
from src.encoding.models import Encoding, DataEncodings, TaskGenerationTypes
from src.encoding.models import ValueEncodings
from src.hyperparameter_optimization.models import HyperparameterOptimizationMethods, HyperparameterOptimization, \
    HyperOptLosses
from src.jobs.job_creator import get_prediction_method_config, set_model_name
from src.jobs.models import Job, JobStatuses, JobTypes
from src.labelling.models import Labelling, LabelTypes, ThresholdTypes
from src.logs.models import Log
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModel, PredictiveModels
from src.split.models import Split, SplitTypes, SplitOrderingMethods

bpi_log_filepath = "cache/log_cache/test_logs/BPI Challenge 2017.xes.gz"
general_example_filepath = 'cache/log_cache/test_logs/general_example.xes'
general_example_filename = 'general_example.xes'
general_example_train_filepath = 'cache/log_cache/test_logs/general_example_train.xes'
general_example_train_filename = 'general_example_train.xes'
general_example_test_filepath_xes = 'cache/log_cache/test_logs/general_example_test.xes'
general_example_test_filepath_csv = 'cache/log_cache/test_logs/general_example_test.csv'
general_example_test_filename = 'general_example_test.xes'
financial_log_filepath = 'cache/log_cache/test_logs/financial_log.xes.gz'
financial_log_filename = 'financial_log.xes.gz'
repair_example_filepath = 'cache/log_cache/test_logs/repair_example.xes'


def create_test_log(log_name: str = 'general_example.xes',
                    log_path: str = 'cache/log_cache/test_logs/general_example.xes') -> Log:
    log = Log.objects.get_or_create(name=log_name, path=log_path)[0]
    return log


def create_test_split(split_type: str = SplitTypes.SPLIT_SINGLE.value,
                      split_ordering_method: str = SplitOrderingMethods.SPLIT_SEQUENTIAL.value,
                      test_size: float = 0.2,
                      original_log: Log = None,
                      train_log: Log = None,
                      test_log: Log = None):
    if split_type == SplitTypes.SPLIT_SINGLE.value:
        if original_log is None:
            original_log = create_test_log()
        split = Split.objects.get_or_create(type=split_type,
                                            original_log=original_log,
                                            test_size=test_size,
                                            splitting_method=split_ordering_method)[0]
    elif split_type == SplitTypes.SPLIT_DOUBLE.value:
        if train_log is None:
            train_log = create_test_log()
        if test_log is None:
            test_log = create_test_log()
        split = Split.objects.get_or_create(type=split_type,
                                            train_log=train_log,
                                            test_log=test_log)[0]
    else:
        raise ValueError('split_type {} not recognized'.format(split_type))
    return split


def create_test_encoding(prefix_length: int = 1,
                         padding: bool = False,
                         value_encoding: str = ValueEncodings.SIMPLE_INDEX.value,
                         add_elapsed_time: bool = False,
                         add_remaining_time: bool = False,
                         add_resources_used: bool = False,
                         add_new_traces: bool = False,
                         add_executed_events: bool = False,
                         task_generation_type: str = TaskGenerationTypes.ONLY_THIS.value) -> Encoding:
    encoding = Encoding.objects.get_or_create(
        data_encoding=DataEncodings.LABEL_ENCODER.value,
        value_encoding=value_encoding,
        prefix_length=prefix_length,
        padding=padding,
        add_elapsed_time=add_elapsed_time,
        add_executed_events=add_executed_events,
        add_remaining_time=add_remaining_time,
        add_new_traces=add_new_traces,
        add_resources_used=add_resources_used,
        task_generation_type=task_generation_type)[0]
    return encoding


def create_test_labelling(label_type: str = LabelTypes.NEXT_ACTIVITY.value,
                          attribute_name: str = None,
                          threshold_type: str = ThresholdTypes.THRESHOLD_MEAN.value,
                          threshold: float = 0.0) -> Labelling:
    labelling = Labelling.objects.get_or_create(
        type=label_type,
        attribute_name=attribute_name,
        threshold_type=threshold_type,
        threshold=threshold
    )[0]
    return labelling


def create_test_clustering(clustering_type: str = ClusteringMethods.NO_CLUSTER.value,
                           configuration: dict = {}) -> Clustering:
    clustering = Clustering.init(clustering_type, configuration)
    return clustering


def create_test_predictive_model(predictive_model: str = PredictiveModels.CLASSIFICATION.value,
                                 prediction_method: str = ClassificationMethods.RANDOM_FOREST.value,
                                 configuration: dict = {}) -> PredictiveModel:
    pred_model = PredictiveModel.init(get_prediction_method_config(predictive_model, prediction_method, configuration))
    return pred_model


def create_test_hyperparameter_optimizer(hyperoptim_type: str = HyperparameterOptimizationMethods.HYPEROPT.value,
                                         performance_metric: HyperOptLosses = HyperOptLosses.ACC.value,
                                         max_evals: int = 10):
    hyperparameter_optimization = HyperparameterOptimization.init({'type': hyperoptim_type,
                                                                   'performance_metric': performance_metric,
                                                                   'max_evals': max_evals})
    return hyperparameter_optimization


def create_test_job(split: Split = None,
                    encoding: Encoding = None,
                    labelling: Labelling = None,
                    clustering: Clustering = None,
                    create_models: bool = False,
                    predictive_model: PredictiveModel = None,
                    job_type=JobTypes.PREDICTION.value,
                    hyperparameter_optimizer: HyperparameterOptimization = None,
                    incremental_train : Job = None):
    job, _ = Job.objects.get_or_create(
        status=JobStatuses.CREATED.value,
        type=job_type,
        split=split if split is not None else create_test_split(),
        encoding=encoding if encoding is not None else create_test_encoding(),
        labelling=labelling if labelling is not None else create_test_labelling(),
        clustering=clustering if clustering is not None else create_test_clustering(),
        create_models=create_models,
        case_id=[1, 2, 3],
        predictive_model=predictive_model if predictive_model is not None else create_test_predictive_model(),
        evaluation=None,
        hyperparameter_optimizer=hyperparameter_optimizer,
        incremental_train=incremental_train
    )
    set_model_name(job)
    return job
