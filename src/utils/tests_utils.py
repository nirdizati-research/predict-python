from src.clustering.models import Clustering, ClusteringMethods
from src.encoding.models import Encoding, DataEncodings, TaskGenerationTypes
from src.hyperparameter_optimization.models import HyperparameterOptimizationMethods, HyperparameterOptimization
from src.jobs.models import Job, JobStatuses, JobTypes
from src.labelling.models import Labelling, LabelTypes, ThresholdTypes
from src.logs.models import Log
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModel, PredictionTypes
from src.split.models import Split, SplitTypes

bpi_log_filepath = "cache/log_cache/test_logs/BPI Challenge 2017.xes.gz"
general_example_filepath = 'cache/log_cache/test_logs/general_example.xes'
general_example_train_filepath = 'cache/log_cache/test_logs/general_example_train.xes'
general_example_test_filepath = 'cache/log_cache/test_logs/general_example_test.xes'
financial_log_filepath = 'cache/log_cache/test_logs/financial_log.xes.gz'
repair_example_filepath = 'cache/log_cache/test_logs/repair_example.xes'


def create_test_log(log_name: str = 'general_example.xes',
                    log_path: str = 'cache/log_cache/test_logs/general_example.xes') -> Log:
    log = Log.objects.get_or_create(name=log_name, path=log_path)[0]
    print('log:', log)
    return log


def create_test_split(split_type: str = SplitTypes.SPLIT_SINGLE.value, log: Log = create_test_log()):
    split = Split.objects.get_or_create(type=split_type, original_log=log)[0]
    print('split:', split)
    return split


def create_test_encoding(prefix_length: int = 1, padding: bool = False,
                         task_generation_type: str = TaskGenerationTypes.ONLY_THIS.value) -> Encoding:
    encoding = Encoding.objects.get_or_create(
        data_encoding=DataEncodings.LABEL_ENCODER.value,
        prefix_length=prefix_length,
        padding=padding,
        task_generation_type=task_generation_type)[0]
    print('encoding:', encoding)
    return encoding


def create_test_labelling(label_type: str = LabelTypes.NEXT_ACTIVITY.value,
                          attribute_name: str = 'label',
                          threshold_type: str = ThresholdTypes.THRESHOLD_MEAN.value,
                          threshold: float = 0.0) -> Labelling:
    labelling = Labelling.objects.get_or_create(
        type=label_type,
        attribute_name=attribute_name,
        threshold_type=threshold_type,
        threshold=threshold
    )[0]
    print('labelling:', labelling)
    return labelling


def create_test_clustering(clustering_type: str = ClusteringMethods.NO_CLUSTER.value,
                           configuration: dict = None) -> Clustering:
    clustering = Clustering.init(clustering_type, configuration)[0]
    print('clustering:', clustering)
    return clustering


def create_test_predictive_model(predictive_model: str = PredictionTypes.CLASSIFICATION.value,
                                 prediction_method: str = ClassificationMethods.RANDOM_FOREST.value) -> PredictiveModel:
    predictive_model = PredictiveModel.init({
        'predictive_model': predictive_model,
        'prediction_method': prediction_method
    })
    print('predictive_model:', predictive_model)
    return predictive_model


def create_test_hyperparameter_optimizer(hyperoptim_type: str = HyperparameterOptimizationMethods.HYPEROPT.value):
    hyperparameter_optimization = HyperparameterOptimization.init({'type': hyperoptim_type})[0]
    print('hyperparameter_optimization:', hyperparameter_optimization)
    return hyperparameter_optimization


def create_test_job(split: Split = create_test_split(),
                    encoding: Encoding = create_test_encoding(),
                    labelling: Labelling = create_test_labelling(),
                    clustering: Clustering = create_test_clustering(),
                    predictive_model: PredictiveModel = create_test_predictive_model(),
                    job_type=JobTypes.PREDICTION.value,
                    hyperparameter_optimizer: HyperparameterOptimization = None):
    job = Job.objects.create(
        status=JobStatuses.CREATED.value,
        type=job_type,
        split=split,
        encoding=encoding,
        labelling=labelling,
        clustering=clustering,
        predictive_model=predictive_model,
        evaluation=None,
        hyperparameter_optimizer=hyperparameter_optimizer
    )
    print('job:', job)
    return job
