import django
django.setup()

import json
import time

from enum import Enum

from src.encoding.models import ValueEncodings, TaskGenerationTypes
from src.hyperparameter_optimization.models import HyperOptLosses, HyperOptAlgorithms, HyperparameterOptimizationMethods
from src.labelling.models import LabelTypes
from src.predictive_model.classification.models import ClassificationMethods
from src.clustering.models import ClusteringMethods
from src.jobs.models import JobStatuses, JobTypes
from src.utils.experiments_utils import upload_split, send_job_request, create_payload, retrieve_job


def retrieve_predictive_model_configuration(config):
    if len(config) == 1:
        config = config[0]['config']
    elif len(config) > 1:
        print('duplicate config')
        config = config[0]['config']
    else:
        print('missing conf')
        return {}
    predictive_model_config = config['predictive_model']
    del predictive_model_config['model_path']
    predictive_model = predictive_model_config['predictive_model']
    del predictive_model_config['predictive_model']
    prediction_method = predictive_model_config['prediction_method']
    del predictive_model_config['prediction_method']
    return {predictive_model + '.' + prediction_method: predictive_model_config}


def init_database(experimentation_type, splits, dataset, base_folder):
    if dataset not in splits:
        splits[dataset] = {}

    if experimentation_type == ExperimentationType.STD.value:
        splits[dataset]['0-40_80-100'] = upload_split(train=base_folder + dataset + '0-40.xes',
                                                      test=base_folder + dataset + '80-100.xes')

        splits[dataset]['0-80_80-100'] = upload_split(train=base_folder + dataset + '0-80.xes',
                                                      test=base_folder + dataset + '80-100.xes')

    elif experimentation_type == ExperimentationType.INCREMENTAL.value:
        splits[dataset]['40-80_80-100'] = upload_split(train=base_folder + dataset + '40-80.xes',
                                                       test=base_folder + dataset + '80-100.xes')

    elif experimentation_type == ExperimentationType.DRIFT_SIZE.value:
        splits[dataset]['40-55_80-100'] = upload_split(train=base_folder + dataset + '40-55.xes',
                                                       test=base_folder + dataset + '80-100.xes')
        splits[dataset]['0-55_80-100'] = upload_split(train=base_folder + dataset + '0-55.xes',
                                                      test=base_folder + dataset + '80-100.xes')


def get_pretrained_model_id(data, prefix, attribute_name, classification_method, dataset, encoding_method):
    model_id = data.loc[
        (data['predictive_model'] == classification_method) &
        (data['encoding_value_encoding'] == encoding_method) &
        (data['encoding_prefix_length'] == prefix) &
        (data['labelling_attribute_name'] == attribute_name) &
        (data['split_id'] == dataset)
        ].filter(items=['predictive_model_id']).values[0][0]
    return model_id


def std_experiments(dataset, prefix_length, models, splits, classification_method, encoding_method):
    models[dataset]['0-40_80-100'] = send_job_request(
        payload=create_payload(
            split=splits[dataset]['0-40_80-100'],
            encodings=[encoding_method],
            encoding={"padding": "zero_padding",
                      "generation_type": TaskGenerationTypes.ALL_IN_ONE.value,
                      "prefix_length": prefix_length,
                      "features": []},
            labeling={"type": LabelTypes.ATTRIBUTE_STRING.value,
                      "attribute_name": "label",
                      "add_remaining_time": False,
                      "add_elapsed_time": False,
                      "add_executed_events": False,
                      "add_resources_used": False,
                      "add_new_traces": False},
            hyperparameter_optimization={"type": HyperparameterOptimizationMethods.HYPEROPT.value,
                                         "max_evaluations": 1000,
                                         "performance_metric": HyperOptLosses.AUC.value,
                                         "algorithm_type": HyperOptAlgorithms.TPE.value},
            classification=[classification_method]
        )
    )[0]['id']

    models[dataset]['0-80_80-100'] = send_job_request(
        payload=create_payload(
            split=splits[dataset]['0-80_80-100'],
            encodings=[encoding_method],
            encoding={"padding": "zero_padding",
                      "generation_type": TaskGenerationTypes.ALL_IN_ONE.value,
                      "prefix_length": prefix_length,
                      "features": []},
            labeling={"type": LabelTypes.ATTRIBUTE_STRING.value,
                      "attribute_name": "label",
                      "add_remaining_time": False,
                      "add_elapsed_time": False,
                      "add_executed_events": False,
                      "add_resources_used": False,
                      "add_new_traces": False},
            hyperparameter_optimization={"type": HyperparameterOptimizationMethods.HYPEROPT.value,
                                         "max_evaluations": 1000,
                                         "performance_metric": HyperOptLosses.AUC.value,
                                         "algorithm_type": HyperOptAlgorithms.TPE.value},
            classification=[classification_method]
        )
    )[0]['id']


def incremental_experiments(dataset, prefix_length, models, splits, classification_method, encoding_method):
    pretrained_model_parameters = retrieve_predictive_model_configuration(
        retrieve_job(config={
            'type': JobTypes.PREDICTION.value,
            # 'status': JobStatuses.COMPLETED.value, # TODO sometimes some jobs hang in running while they are actually finished
            'create_models': True,
            'split': splits[dataset]['0-40_80-100'],
            'encoding': {"value_encoding": encoding_method,
                         "padding": True,
                         "task_generation_type": TaskGenerationTypes.ALL_IN_ONE.value,
                         "prefix_length": prefix_length},
            'labelling': {"type": LabelTypes.ATTRIBUTE_STRING.value,
                          "attribute_name": "label",
                          "add_remaining_time": False,
                          "add_elapsed_time": False,
                          "add_executed_events": False,
                          "add_resources_used": False,
                          "add_new_traces": False},
            'hyperparameter_optimization': {"optimization_method": HyperparameterOptimizationMethods.HYPEROPT.value},
                                            # "max_evaluations": 1000, #TODO not yet supported
                                            # "performance_metric": HyperOptLosses.AUC.value,
                                            # "algorithm_type": HyperOptAlgorithms.TPE.value},
            'predictive_model': {'predictive_model': 'classification',
                                 'prediction_method': classification_method},
            'clustering': {'clustering_method': ClusteringMethods.NO_CLUSTER.value}
        })
    )

    payload = create_payload(
        split=splits[dataset]['0-80_80-100'],
        encodings=[encoding_method],
        encoding={"padding": "zero_padding",
                  "generation_type": TaskGenerationTypes.ALL_IN_ONE.value,
                  "prefix_length": prefix_length,
                  "features": []},
        labeling={"type": LabelTypes.ATTRIBUTE_STRING.value,
                  "attribute_name": "label",
                  "add_remaining_time": False,
                  "add_elapsed_time": False,
                  "add_executed_events": False,
                  "add_resources_used": False,
                  "add_new_traces": False},
        hyperparameter_optimization={"type": HyperparameterOptimizationMethods.NONE.value},
        classification=[classification_method]
    )
    payload.update(pretrained_model_parameters)
    models[dataset]['0-80_80-100'] = send_job_request(payload=payload)[0]['id']

    if classification_method != ClassificationMethods.RANDOM_FOREST.value:
        payload = create_payload(
                split=splits[dataset]['40-80_80-100'],
                encodings=[encoding_method],
                encoding={"padding": "zero_padding",
                          "generation_type": TaskGenerationTypes.ALL_IN_ONE.value,
                          "prefix_length": prefix_length,
                          "features": []},
                labeling={"type": LabelTypes.ATTRIBUTE_STRING.value,
                          "attribute_name": "label",
                          "add_remaining_time": False,
                          "add_elapsed_time": False,
                          "add_executed_events": False,
                          "add_resources_used": False,
                          "add_new_traces": False},
                classification=[classification_method],
                hyperparameter_optimization={"type": HyperparameterOptimizationMethods.NONE.value}
        )
        payload.update(pretrained_model_parameters)
        models[dataset]['40-80_80-100'] = send_job_request(payload=payload)[0]['id']


def drift_size_experimentation(dataset, prefix_length, models, splits, classification_method, encoding_method):
    if classification_method != "randomForest":
        models[dataset]['40-55_80-100'] = send_job_request(
            payload=create_payload(
                split=splits[dataset]['40-55_80-100'],
                encodings=[encoding_method],
                encoding={"padding": "zero_padding",
                          "generation_type": TaskGenerationTypes.ALL_IN_ONE.value,
                          "prefix_length": prefix_length,
                          "features": []},
                labeling={"type": LabelTypes.ATTRIBUTE_STRING.value,
                          "attribute_name": "label",
                          "add_remaining_time": False,
                          "add_elapsed_time": False,
                          "add_executed_events": False,
                          "add_resources_used": False,
                          "add_new_traces": False},
                classification=[classification_method],
                hyperparameter_optimization={"type": HyperparameterOptimizationMethods.NONE.value},
                incremental_train={
                    "base_model": get_pretrained_model_id(data, prefix_length, 'label', classification_method,
                                                          splits[dataset]['0-40_80-100'], encoding_method)}
                # MODEL_HYPERPARAMETERS={
                #     'classification_' + classification_method: Job.objects.filter(id=models[dataset]['0-40_80-100'])[0].predictive_model.to_dict()
                # }
            )
        )[0]['id']

    models[dataset]['0-55_80-100'] = send_job_request(
        payload=create_payload(
            split=splits[dataset]['0-55_80-100'],
            encodings=[encoding_method],
            encoding={"padding": "zero_padding",
                      "generation_type": TaskGenerationTypes.ALL_IN_ONE.value,
                      "prefix_length": prefix_length,
                      "features": []},
            labeling={"type": LabelTypes.ATTRIBUTE_STRING.value,
                      "attribute_name": "label",
                      "add_remaining_time": False,
                      "add_elapsed_time": False,
                      "add_executed_events": False,
                      "add_resources_used": False,
                      "add_new_traces": False},
            classification=[classification_method]
        )
    )[0]['id']


class ExperimentationType(Enum):
    STD = 'std'
    INCREMENTAL = 'incremental'
    DRIFT_SIZE = 'drift_size'


def launch_experimentation(experimentation_type, datasets, splits, base_folder, models, prefixes=[10, 30, 50, 70],
                           classification_methods=[ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value],
                           encodings=[ValueEncodings.SIMPLE_INDEX.value]):
    for dataset in datasets:
        init_database(experimentation_type, splits, dataset, base_folder)

        print(dataset, '[:::] Batch of logs uploaded')
        if dataset not in models:
            models[dataset] = {}

        for prefix_length in prefixes:  # NB: if you add something the splits and models are overwritten
            for classification_method in classification_methods:  # NB: if you add something the models are overwritten
                for encoding_method in encodings:  # NB: if you add something the models are overwritten
                    if experimentation_type == ExperimentationType.STD.value:
                        std_experiments(dataset, prefix_length, models, splits, classification_method, encoding_method)
                    elif experimentation_type == ExperimentationType.INCREMENTAL.value:
                        incremental_experiments(dataset, prefix_length, models, splits, classification_method,
                                                encoding_method)
                    elif experimentation_type == ExperimentationType.DRIFT_SIZE.value:
                        drift_size_experimentation(dataset, prefix_length, models, splits, classification_method,
                                                   encoding_method)
            print(dataset, '[:::] Batch of tasks created')
        time.sleep(180)


if __name__ == '__main__':
    print("Starting experiments")

    base_folder = '/home/wrizzi/Documents/datasets/'
    base_folder = '/Users/Brisingr/Desktop/TEMP/dataset/prom_labeled_data/CAiSE18/'

    experimentation = ExperimentationType.INCREMENTAL.value

    datasets1 = [
        'BPI11/f1/',
        'BPI11/f2/',
        'BPI11/f3/',
        'BPI11/f4/',
        'BPI15/f1/',
        'BPI15/f2/',
        'BPI15/f3/'
    ]

    datasets2 = [
        'Drift1/f1/',
        'Drift2/f1/'
    ]

    split_sizes = [
        '0-40.xes',
        '0-60.xes',
        '0-55.xes',
        '0-80.xes',
        '40-80.xes',
        '40-60.xes',
        '40-55.xes',
        '80-100.xes'
    ]

    # TODO load from memory
    splits = {
        'BPI11/f1/': {
            '0-40_80-100': 8,
            '0-80_80-100': 9,
            '40-80_80-100': 1111,
        },
        'BPI11/f2/': {
            '0-40_80-100': 10,
            '0-80_80-100': 11,
            '40-80_80-100': 1111,
        },
        'BPI11/f3/': {
            '0-40_80-100': 12,
            '0-80_80-100': 13,
            '40-80_80-100': 1111,
        },
        'BPI11/f4/': {
            '0-40_80-100': 14,
            '0-80_80-100': 15,
            '40-80_80-100': 1111,
        },
        'BPI15/f1/': {
            '0-40_80-100': 16,
            '0-80_80-100': 17,
            '40-80_80-100': 1111,
        },
        'BPI15/f2/': {
            '0-40_80-100': 18,
            '0-80_80-100': 19,
            '40-80_80-100': 1111,
        },
        'BPI15/f3/': {
            '0-40_80-100': 20,
            '0-80_80-100': 21,
            '40-80_80-100': 1111,
        },
        'Drift1/f1/': {
            '0-40_80-100': 22,
            '0-80_80-100': 23,
            '40-80_80-100': 1111,

            '40-60_80-100': 1111,
            '0-60_80-100': 1111,
            '40-55_80-100': 1111,
            '0-55_80-100': 1111
        },
        'Drift2/f1/': {
            '0-40_80-100': 24,
            '0-80_80-100': 25,
            '40-80_80-100': 1111,

            '40-60_80-100': 1111,
            '0-60_80-100': 1111,
            '40-55_80-100': 1111,
            '0-55_80-100': 1111
        }
    }

    models = {}
    if experimentation == ExperimentationType.STD.value:
        launch_experimentation(
            ExperimentationType.STD.value,
            datasets1,
            splits,
            base_folder,
            models,
            prefixes=[30, 50, 70],
            classification_methods=[
                ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value,
                ClassificationMethods.SGDCLASSIFIER.value,
                ClassificationMethods.PERCEPTRON.value,
                ClassificationMethods.RANDOM_FOREST.value],
            encodings=[
                ValueEncodings.SIMPLE_INDEX.value,
                ValueEncodings.COMPLEX.value]
        )
        # json.dump(splits, open("splits.json", 'w'))
        # json.dump(models, open("models.json", 'w'))
    elif experimentation == ExperimentationType.DRIFT_SIZE.value:
        launch_experimentation(
            ExperimentationType.DRIFT_SIZE.value,
            datasets2,
            splits,
            base_folder,
            models,
            prefixes=[3, 5, 7],
            classification_methods=[
                ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value,
                ClassificationMethods.SGDCLASSIFIER.value,
                ClassificationMethods.PERCEPTRON.value,
                ClassificationMethods.RANDOM_FOREST.value],
            encodings=[
                ValueEncodings.SIMPLE_INDEX.value,
                ValueEncodings.COMPLEX.value]
        )
        # json.dump(splits, open("splits.json", 'w'))
        # json.dump(models, open("models.json", 'w'))
    elif experimentation == ExperimentationType.INCREMENTAL.value:
        # splits = json.load(open("../splits.json", 'r'))
        # models = json.load(open("../models.json", 'r'))

        launch_experimentation(
            ExperimentationType.INCREMENTAL.value,
            datasets1,
            splits,
            base_folder,
            models,
            prefixes=[30, 50, 70],
            classification_methods=[
                ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value,
                ClassificationMethods.SGDCLASSIFIER.value,
                ClassificationMethods.PERCEPTRON.value,
                ClassificationMethods.RANDOM_FOREST.value],
            encodings=[
                ValueEncodings.SIMPLE_INDEX.value,
                ValueEncodings.COMPLEX.value]
        )
        json.dump(splits, open("splits.json", 'w'))
        json.dump(models, open("models.json", 'w'))

    print("End of the experiments")
