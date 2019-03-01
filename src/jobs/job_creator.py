from src.clustering.models import Clustering

from src.clustering.methods_default_config import clustering_kmeans
from src.clustering.models import ClusteringMethods
from src.core.common import CONF_MAP, ALL_CONFIGS
from src.encoding.encoding_container import UP_TO
from src.encoding.models import Encoding
from src.jobs.models import Job, JobStatuses, JobTypes
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel
from src.predictive_model.models import PredictiveModelTypes


def generate(split, payload, generation_type=PredictiveModelTypes.CLASSIFICATION):
    jobs = []

    config = payload['config']
    label = config['label']

    for method in config['methods']:
        for clustering in config['clusterings']:
            for encMethod in config['encodings']:
                encoding = config['encoding']
                if encoding['generation_type'] == UP_TO:
                    for i in range(1, encoding['prefix_length'] + 1):
                        item = Job.objects.get_or_create(
                            status=JobStatuses.CREATED.value,
                            type=generation_type,

                            split=split,
                            encoding=Encoding.objects.get_or_create(
                                data_encoding=encMethod,
                                # TODO: @HitLuca [value_encoding=,]
                                additional_features=label['add_remaining_time'] or label['add_elapsed_time'] or label['add_executed_events'] or label['add_resources_used'] or label['add_new_traces'],
                                temporal_features=label['add_remaining_time'] or label['add_elapsed_time'],
                                intercase_features=label['add_executed_events'] or label['add_resources_used'] or label['add_new_traces'],
                                prefix_len=i,
                                padding=config['encoding']['padding']
                            ),
                            labelling=Labelling.objects.get_or_create(
                                type=label['type'],
                                attribute_name=label['attribute_name'],
                                threshold_type=label['threshold_type'],
                                threshold=label['threshold']
                            ),
                            clustering=Clustering.init(clustering, configuration=None),
                            predictive_model=PredictiveModel.init(payload['type'], configuration=payload),
                        )
                        jobs.append(item)
                else:
                    item = Job.objects.get_or_create(
                        status=JobStatuses.CREATED.value,
                        type=generation_type,

                        split=split,
                        encoding=Encoding.objects.get_or_create(
                            data_encoding=encMethod,
                            # TODO: @HitLuca [value_encoding=,]
                            additional_features=label['add_remaining_time'] or label['add_elapsed_time'] or label['add_executed_events'] or label['add_resources_used'] or label['add_new_traces'],
                            temporal_features=label['add_remaining_time'] or label['add_elapsed_time'],
                            intercase_features=label['add_executed_events'] or label['add_resources_used'] or label['add_new_traces'],
                            prefix_len=config['encoding']['prefix_length'],
                            padding=config['encoding']['padding']
                        ),
                        labelling=Labelling.objects.get_or_create(
                            type=label['type'],
                            attribute_name=label['attribute_name'],
                            threshold_type=label['threshold_type'],
                            threshold=label['threshold']
                        ),
                        clustering=Clustering.init(clustering, configuration=None),
                        predictive_model=PredictiveModel.init(payload['type'], configuration=payload)
                    )
                    jobs.append(item)

    return jobs


def generate_labelling(split, payload):
    jobs = []
    encoding = payload['config']['encoding']
    if encoding['generation_type'] == UP_TO:
        for i in range(1, encoding['prefix_length'] + 1):
            item = Job.objects.get_or_create(
                status=JobStatuses.CREATED.value,
                type=JobTypes.LABELLING.value,

                split=split,
                encoding=Encoding.objects.get_or_create(
                    data_encoding='label_encoder',
                    #TODO: @HitLuca [value_encoding=,]
                    additional_features=payload['label']['add_remaining_time'] or payload['label']['add_elapsed_time'] or
                                        payload['label']['add_executed_events'] or payload['label']['add_resources_used'] or
                                        payload['label']['add_new_traces'],
                    temporal_features=payload['label']['add_remaining_time'] or payload['label']['add_elapsed_time'],
                    intercase_features= payload['label']['add_executed_events'] or payload['label']['add_resources_used'] or
                                        payload['label']['add_new_traces'],
                    prefix_len=i,
                    padding=payload['encoding']['padding']
                ),
                labelling=Labelling.objects.get_or_create(
                    type=payload['label']['type'],
                    attribute_name=payload['label']['attribute_name'],
                    threshold_type=payload['label']['threshold_type'],
                    threshold=payload['label']['threshold']
                )
            )
            jobs.append(item)
    else:
        item = Job.objects.get_or_create(
            status=JobStatuses.CREATED.value,
            type=JobTypes.LABELLING.value,

            split=split,
            encoding=Encoding.objects.get_or_create(
                data_encoding='label_encoder',
                # TODO: @HitLuca [value_encoding=,]
                additional_features=payload['label']['add_remaining_time'] or payload['label']['add_elapsed_time'] or
                                    payload['label']['add_executed_events'] or payload['label']['add_resources_used'] or
                                    payload['label']['add_new_traces'],
                temporal_features=payload['label']['add_remaining_time'] or payload['label']['add_elapsed_time'],
                intercase_features=payload['label']['add_executed_events'] or payload['label']['add_resources_used'] or
                                   payload['label']['add_new_traces'],
                prefix_len=payload['encoding']['prefix_length'],
                padding=payload['encoding']['padding']
            ),
            labelling=Labelling.objects.get_or_create(
                type=payload['label']['type'],
                attribute_name=payload['label']['attribute_name'],
                threshold_type=payload['label']['threshold_type'],
                threshold=payload['label']['threshold']
            )
        )
        jobs.append(item)

    return jobs


def update(split, payload):  # TODO adapt to allow selecting the predictive_model to update
    jobs = []
    for method in payload['config']['methods']:
        for clustering in payload['config']['clusterings']:
            for encMethod in payload['config']['encodings']:
                encoding = payload['config']['encoding']
                if encoding['generation_type'] == UP_TO:
                    for i in range(1, encoding['prefix_length'] + 1):
                        item = Job.objects.get_or_create(
                            status=JobStatuses.CREATED.value,
                            type=payload['type'],

                            split=split,
                            encoding=Encoding.objects.get_or_create(
                                data_encoding=encMethod,
                                # TODO: @HitLuca [value_encoding=,]
                                additional_features=payload['label']['add_remaining_time'] or payload['label'][
                                    'add_elapsed_time'] or
                                                    payload['label']['add_executed_events'] or payload['label'][
                                                        'add_resources_used'] or
                                                    payload['label']['add_new_traces'],
                                temporal_features=payload['label']['add_remaining_time'] or payload['label'][
                                    'add_elapsed_time'],
                                intercase_features=payload['label']['add_executed_events'] or payload['label'][
                                    'add_resources_used'] or
                                                   payload['label']['add_new_traces'],
                                prefix_len=i,
                                padding=payload['encoding']['padding']
                            ),
                            labelling=Labelling.objects.get_or_create(
                                type=payload['label']['type'],
                                attribute_name=payload['label']['attribute_name'],
                                threshold_type=payload['label']['threshold_type'],
                                threshold=payload['label']['threshold']
                            ),
                            clustering=Clustering.init(clustering, configuration=None),
                            predictive_model=PredictiveModel.init(payload['type'], configuration=payload)
                        )
                        jobs.append(item)
                else:
                    item = Job.objects.get_or_create(
                        status=JobStatuses.CREATED.value,
                        type=payload['type'],

                        split=split,
                        encoding=Encoding.objects.get_or_create(
                            data_encoding='label_encoder',
                            # TODO: @HitLuca [value_encoding=,]
                            additional_features=payload['label']['add_remaining_time'] or payload['label'][
                                'add_elapsed_time'] or
                                                payload['label']['add_executed_events'] or payload['label'][
                                                    'add_resources_used'] or
                                                payload['label']['add_new_traces'],
                            temporal_features=payload['label']['add_remaining_time'] or payload['label'][
                                'add_elapsed_time'],
                            intercase_features=payload['label']['add_executed_events'] or payload['label'][
                                'add_resources_used'] or
                                               payload['label']['add_new_traces'],
                            prefix_len=payload['encoding']['prefix_length'],
                            padding=payload['encoding']['padding']
                        ),
                        labelling=Labelling.objects.get_or_create(
                            type=payload['label']['type'],
                            attribute_name=payload['label']['attribute_name'],
                            threshold_type=payload['label']['threshold_type'],
                            threshold=payload['label']['threshold']
                        ),
                        clustering=Clustering.init(clustering, configuration=None),
                        predictive_model=PredictiveModel.init(payload['type'], configuration=payload)
                    )
                    jobs.append(item)
    return jobs


# def create_config(payload: dict, enc_method: str, clustering: str, method: str, prefix_length: int):
#     """Turn lists to single values"""
#     config = dict(payload['config'])
#     del config['encodings']
#     del config['clusterings']
#     del config['methods']
#
#     # Extract and merge configurations
#     method_conf_name = "{}.{}".format(payload['type'], method)
#
#     method_conf = {**CONF_MAP[method_conf_name](), **payload['config'].get(method_conf_name, dict())}
#     # Remove configs that are not needed for this method
#     for any_conf_name in ALL_CONFIGS:
#         try:
#             del config[any_conf_name]
#         except KeyError:
#             pass
#
#     if clustering == ClusteringMethods.KMEANS:
#         config['kmeans'] = {**clustering_kmeans(), **payload['config'].get('kmeans', dict())}
#     elif 'kmeans' in config:
#         del config['kmeans']
#     config[method_conf_name] = method_conf
#     config['clustering'] = clustering
#     config['method'] = method
#     # Encoding stuff rewrite
#     config['encoding']['method'] = enc_method
#     config['encoding']['prefix_length'] = prefix_length
#     return config
#
#
# def create_config_labelling(payload: dict, prefix_length: int):
#     """For labelling job"""
#     config = dict(payload['config'])
#
#     # All methods are the same, so defaulting to SIMPLE_INDEX
#     # Remove when encoding and labelling are separated
#     config['encoding']['method'] = DataEncodings.SIMPLE_INDEX
#     config['encoding']['prefix_length'] = prefix_length
#     return config
