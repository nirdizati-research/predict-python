from src.clustering.models import Clustering, ClusteringMethods
from src.encoding.models import Encoding, DataEncodings
from src.jobs.models import Job, JobStatuses, JobTypes
from src.labelling.models import Labelling, LabelTypes, ThresholdTypes
from src.logs.models import Log
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModel, PredictiveModelTypes
from src.split.models import Split, SplitTypes

bpi_log_filepath = "cache/log_cache/test_logs/BPI Challenge 2017.xes.gz"
general_example_filepath = 'cache/log_cache/test_logs/general_example.xes'
general_example_train_filepath = 'cache/log_cache/test_logs/general_example_train.xes'
general_example_test_filepath = 'cache/log_cache/test_logs/general_example_test.xes'
financial_log_filepath = 'cache/log_cache/test_logs/financial_log.xes.gz'
repair_example_filepath = 'cache/log_cache/test_logs/repair_example.xes'


def create_test_log(log_name: str = 'general_example.xes', log_filepath: str = general_example_filepath) -> Log:
    log = Log.objects.get_or_create(name=log_name, path=log_filepath)
    return log[0]


def create_test_split(split_type: str = SplitTypes.SPLIT_SINGLE, log: Log = create_test_log()):
    split = Split.objects.get_or_create(type=split_type, original_log=log)
    return split[0]


def create_test_encoding(prefix_length: int = 1, padding: bool = False) -> Encoding:
    encoding = Encoding.objects.get_or_create(
        data_encoding=DataEncodings.LABEL_ENCODER,
        prefix_len=prefix_length,
        padding=padding)
    return encoding[0]


def create_test_labelling(label_type: str = LabelTypes.NEXT_ACTIVITY.value,
                          attribute_name: str = 'label',
                          threshold_type: str = ThresholdTypes.THRESHOLD_MEAN.value,
                          threshold: float = 0.0) -> Labelling:
    labelling = Labelling.objects.get_or_create(
        type=label_type,
        attribute_name=attribute_name,
        threshold_type=threshold_type,
        threshold=threshold
    )
    return labelling[0]


def create_test_clustering(clustering_type: str = ClusteringMethods.NO_CLUSTER.value,
                           configuration: dict = None) -> Clustering:
    clustering = Clustering.init(clustering_type, configuration)
    return clustering[0]


def create_test_predictive_model(prediction_type: str = PredictiveModelTypes.CLASSIFICATION.value,
                                 predictive_model_type: str = ClassificationMethods.RANDOM_FOREST.value) -> PredictiveModel:
    predictive_model = PredictiveModel.init(prediction_type, {'type': predictive_model_type})
    return predictive_model[0]


def create_test_job(split: Split = create_test_split(),
                    encoding: Encoding = create_test_encoding(),
                    labelling: Labelling = create_test_labelling(),
                    clustering: Clustering = create_test_clustering(),
                    predictive_model: PredictiveModel = create_test_predictive_model(),
                    job_type=JobTypes.PREDICTION.value):
    job = Job.objects.create(
        status=JobStatuses.CREATED.value,
        type=job_type,
        split=split,
        encoding=encoding,
        labelling=labelling,
        clustering=clustering,
        predictive_model=predictive_model,
        evaluation=None
    )
    return job
