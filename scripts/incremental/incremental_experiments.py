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
from src.utils.experiments_utils import upload_split, send_job_request, create_classification_payload, retrieve_job


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
                                                      test=base_folder + dataset + '80-100.xes', server_name='ashkin', server_port='50401')

        splits[dataset]['0-80_80-100'] = upload_split(train=base_folder + dataset + '0-80.xes',
                                                      test=base_folder + dataset + '80-100.xes', server_name='ashkin', server_port='50401')

    elif experimentation_type == ExperimentationType.INCREMENTAL.value:
        splits[dataset]['40-80_80-100'] = upload_split(train=base_folder + dataset + '40-80.xes',
                                                       test=base_folder + dataset + '80-100.xes', server_name='ashkin', server_port='50401')

    elif experimentation_type == ExperimentationType.DRIFT_SIZE.value:
        splits[dataset]['40-55_80-100'] = upload_split(train=base_folder + dataset + '40-55.xes',
                                                       test=base_folder + dataset + '80-100.xes', server_name='ashkin', server_port='50401')
        splits[dataset]['0-55_80-100'] = upload_split(train=base_folder + dataset + '0-55.xes',
                                                      test=base_folder + dataset + '80-100.xes', server_name='ashkin', server_port='50401')


def get_pretrained_model_id(config):
    if len(config) == 1:
        model_id = config[0]['id']
    elif len(config) > 1:
        print('duplicate model')
        model_id = config[0]['id']
    else:
        print('missing model')
        return {}
    return model_id


def std_experiments(dataset, prefix_length, models, splits, classification_method, encoding_method):
    models[dataset]['0-40_80-100'] = send_job_request(
        payload=create_classification_payload(
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
        ), server_port='50401', server_name='ashkin'
    )[0]['id']

    models[dataset]['0-80_80-100'] = send_job_request(
        payload=create_classification_payload(
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
        ), server_port='50401', server_name='ashkin'
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
        }, server_name='ashkin', server_port='50401')
    )

    payload = create_classification_payload(
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
    models[dataset]['0-80_80-100'] = send_job_request(payload=payload, server_port='50401', server_name='ashkin')[0]['id']

    if classification_method != ClassificationMethods.RANDOM_FOREST.value:
        payload = create_classification_payload(
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
                hyperparameter_optimization={"type": HyperparameterOptimizationMethods.NONE.value},
                incremental_train=[
                    get_pretrained_model_id(
                        config=retrieve_job(config={
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
                            'hyperparameter_optimization': {
                                "optimization_method": HyperparameterOptimizationMethods.HYPEROPT.value},
                            # "max_evaluations": 1000, #TODO not yet supported
                            # "performance_metric": HyperOptLosses.AUC.value,
                            # "algorithm_type": HyperOptAlgorithms.TPE.value},
                            'predictive_model': {'predictive_model': 'classification',
                                                 'prediction_method': classification_method},
                            'clustering': {'clustering_method': ClusteringMethods.NO_CLUSTER.value}
                        }, server_name='ashkin', server_port='50401')
                    )
                ]
        )
        payload.update(pretrained_model_parameters)
        models[dataset]['40-80_80-100'] = send_job_request(payload=payload, server_port='50401', server_name='ashkin')[0]['id']


def drift_size_experimentation(dataset, prefix_length, models, splits, classification_method, encoding_method):
    if classification_method != "randomForest":
        models[dataset]['40-55_80-100'] = send_job_request(
            payload=create_classification_payload(
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
                incremental_train=[
                    get_pretrained_model_id(
                        config=retrieve_job(config={
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
                        }, server_name='ashkin', server_port='50401')
                    )
                ]
            ), server_port='50401', server_name='ashkin'
        )[0]['id']

    models[dataset]['0-55_80-100'] = send_job_request(
        payload=create_classification_payload(
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
            classification=[classification_method],
            hyperparameter_optimization={"type": HyperparameterOptimizationMethods.HYPEROPT.value,
                                         "max_evaluations": 1000,
                                         "performance_metric": HyperOptLosses.AUC.value,
                                         "algorithm_type": HyperOptAlgorithms.TPE.value},
        ), server_port='50401', server_name='ashkin'
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
    # base_folder = '/Users/Brisingr/Desktop/TEMP/dataset/prom_labeled_data/CAiSE18/'

    experimentation = ExperimentationType.DRIFT_SIZE.value

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
            '0-40_80-100': 55,
            '0-80_80-100': 56,
            '40-80_80-100': 38,
        },
        'BPI11/f2/': {
            '0-40_80-100': 57,
            '0-80_80-100': 58,
            '40-80_80-100': 39,
        },
        'BPI11/f3/': {
            '0-40_80-100': 59,
            '0-80_80-100': 60,
            '40-80_80-100': 40,
        },
        'BPI11/f4/': {
            '0-40_80-100': 61,
            '0-80_80-100': 62,
            '40-80_80-100': 41,
        },
        'BPI15/f1/': {
            '0-40_80-100': 63,
            '0-80_80-100': 64,
            '40-80_80-100': 42,
        },
        'BPI15/f2/': {
            '0-40_80-100': 65,
            '0-80_80-100': 66,
            '40-80_80-100': 43,
        },
        'BPI15/f3/': {
            '0-40_80-100': 67,
            '0-80_80-100': 68,
            '40-80_80-100': 44,
        },
        'Drift1/f1/': {
            '0-40_80-100': 69,
            '0-80_80-100': 70,
            '40-80_80-100': 45,

            '40-60_80-100': 1111,
            '0-60_80-100': 1111,
            '40-55_80-100': 36,  # +TANTO perche' uno e' stato ciccato
            '0-55_80-100': 1111
        },
        'Drift2/f1/': {
            '0-40_80-100': 71,
            '0-80_80-100': 72,
            '40-80_80-100': 46,

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

        launch_experimentation(
            ExperimentationType.STD.value,
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
        json.dump(splits, open("splits_1.json", 'w'))
        json.dump(models, open("models_1.json", 'w'))
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
        json.dump(splits, open("splits_2.json", 'w'))
        json.dump(models, open("models_2.json", 'w'))
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

        launch_experimentation(
            ExperimentationType.INCREMENTAL.value,
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

        json.dump(splits, open("splits_3.json", 'w'))
        json.dump(models, open("models_3.json", 'w'))

    print("End of the experiments")
