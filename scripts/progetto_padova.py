from pm4py.objects.log.importer.xes.factory import import_log

from src.clustering.models import Clustering, ClusteringMethods
from src.core.core import _init_clusterer, MODEL, ModelActions
from src.encoding.common import encode_label_logs, LabelTypes
from src.encoding.models import Encoding, TaskGenerationTypes, ValueEncodings, DataEncodings
from src.explanation.explanation import explanation
from src.explanation.models import Explanation, ExplanationTypes
from src.hyperparameter_optimization.models import HyperparameterOptimization, HyperparameterOptimizationMethods, \
    HyperOptAlgorithms, HyperOptLosses
from src.jobs.job_creator import get_prediction_method_config
from src.jobs.models import Job, JobStatuses, JobTypes
from src.labelling.models import Labelling
from src.logs.log_service import create_log
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModel, PredictiveModels
from src.split.models import Split, SplitTypes
from src.split.splitting import get_train_test_log

BASE_DIR = 'cache/log_cache/'
RELATIVE_TRAIN_PATH = 'train_set.xes'
RELATIVE_VALIDATION_PATH = 'validation_set.xes'
EXPLANATION_TARGET = 1


def progetto_padova():
    JOB = Job.objects.get_or_create(
        status=JobStatuses.CREATED.value,
        type=JobTypes.PREDICTION.value,
        split=Split.objects.get_or_create(  # this creates the split of the log
            type=SplitTypes.SPLIT_DOUBLE.value,
            train_log=create_log(  # this imports the log
                import_log(BASE_DIR + RELATIVE_TRAIN_PATH),
                RELATIVE_TRAIN_PATH,
                BASE_DIR
            ),
            test_log=create_log(  # this imports the log
                import_log(BASE_DIR + RELATIVE_VALIDATION_PATH),
                RELATIVE_VALIDATION_PATH,
                BASE_DIR
            )
        )[0],
        encoding=Encoding.objects.get_or_create(  # this defines the encoding method
            data_encoding=DataEncodings.LABEL_ENCODER.value,
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=False,
            add_remaining_time=False,
            add_executed_events=False,
            add_resources_used=False,
            add_new_traces=False,
            prefix_length=5,
            padding=True,
            task_generation_type=TaskGenerationTypes.ALL_IN_ONE.value,
            features=[]
        )[0],
        labelling=Labelling.objects.get_or_create(  # this defines the label
            type=LabelTypes.ATTRIBUTE_STRING.value,
            attribute_name='label',
            threshold_type=None,
            threshold=None
        )[0],
        clustering=Clustering.init(ClusteringMethods.NO_CLUSTER.value, configuration={}),
        predictive_model=PredictiveModel.init(  # this defines the predictive model
            get_prediction_method_config(
                PredictiveModels.CLASSIFICATION.value,
                ClassificationMethods.DECISION_TREE.value,
                payload={
                    'max_depth': 2,
                    'min_samples_split': 2,
                    'min_samples_leaf': 2
                }
            )
        ),
        hyperparameter_optimizer=HyperparameterOptimization.init({  # this defines the hyperparameter optimisation procedure
            'type': HyperparameterOptimizationMethods.HYPEROPT.value,
            'max_evaluations': 10,
            'performance_metric': HyperOptAlgorithms.TPE.value,
            'algorithm_type': HyperOptLosses.AUC.value
        }),
        create_models=True
    )[0]

    # encode
    train_log, test_log, additional_columns = get_train_test_log(JOB.split)
    train_df, test_df = encode_label_logs(train_log, test_log, JOB)

    # train + evaluate
    clusterer = _init_clusterer(JOB.clustering, train_df)
    results, model_split = MODEL[JOB.predictive_model.predictive_model][ModelActions.BUILD_MODEL_AND_TEST.value](
        train_df,
        test_df,
        clusterer,
        JOB
    )

    # lime
    exp, _ = Explanation.objects.get_or_create(
        type=ExplanationTypes.LIME.value,
        split=JOB.split,
        predictive_model=JOB.predictive_model,
        job=JOB
    )
    exp.save()

    error, result = explanation(exp.id, int(EXPLANATION_TARGET))

