from src.clustering.models import Clustering
from src.core.constants import PREDICTION, RANDOM_FOREST, CLASSIFICATION, NO_CLUSTER
from src.encoding.encoding_container import LABEL_ENCODER
from src.encoding.models import Encoding
from src.jobs.models import Job, CREATED
from src.labelling.label_container import NEXT_ACTIVITY
from src.labelling.models import Labelling
from src.logs.models import Log
from src.predictive_model.models import PredictiveModel
from src.split.models import Split, SPLIT_SINGLE

bpi_log_filepath = "cache/log_cache/test_logs/BPI Challenge 2017.xes.gz"
general_example_filepath = 'cache/log_cache/test_logs/general_example.xes'
general_example_train_filepath = 'cache/log_cache/test_logs/general_example_train.xes'
general_example_test_filepath = 'cache/log_cache/test_logs/general_example_test.xes'
financial_log_filepath = 'cache/log_cache/test_logs/financial_log.xes.gz'
repair_example_filepath = 'cache/log_cache/test_logs/repair_example.xes'


def create_test_log(log_name: str = 'general_example.xes', log_filepath: str = general_example_filepath) -> Log:
    log = Log.objects.get_or_create(name=log_name, path=log_filepath)
    return log[0]


def create_test_split(split_type: str = SPLIT_SINGLE, log: Log = create_test_log()):
    split = Split.objects.get_or_create(type=split_type, original_log=log)
    return split[0]


def create_test_encoding(prefix_length: int = 1, padding: bool = False) -> Encoding:
    encoding = Encoding.objects.get_or_create(
        data_encoding=LABEL_ENCODER,
        prefix_len=prefix_length,
        padding=padding)
    return encoding[0]


def create_test_labelling(label_type: str = NEXT_ACTIVITY) -> Labelling:
    labelling = Labelling.objects.get_or_create(
        type=label_type)
    return labelling[0]


def create_test_clustering(clustering_type: str = NO_CLUSTER, configuration: dict = None) -> Clustering:
    clustering = Clustering.init(clustering_type, configuration)
    return clustering[0]


def create_test_predictive_model(prediction_type: str = CLASSIFICATION,
                                 predictive_model_type: str = RANDOM_FOREST) -> PredictiveModel:
    predictive_model = PredictiveModel.init(prediction_type, {'type': predictive_model_type})
    return predictive_model[0]


def create_test_job_prediction(split: Split = create_test_split(),
                               encoding: Encoding = create_test_encoding(),
                               labelling: Labelling = create_test_labelling(),
                               clustering: Clustering = create_test_clustering(),
                               predictive_model: PredictiveModel = create_test_predictive_model()):
    job = Job.objects.create(
        status=CREATED,
        type=PREDICTION,
        split=split,
        encoding=encoding,
        labelling=labelling,
        clustering=clustering,
        predictive_model=predictive_model,
        evaluation=None
    )
    return job
