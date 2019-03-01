from copy import deepcopy

from src.clustering.models import Clustering
from src.core.constants import KMEANS, ALL_CONFIGS, LABELLING, CLASSIFICATION, PREDICTION
from src.core.default_configuration import CONF_MAP, clustering_kmeans
from src.encoding.encoding_container import UP_TO, SIMPLE_INDEX
from src.encoding.models import Encoding
from src.jobs.models import Job, CREATED
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel


def generate(split, payload, generation_type=PREDICTION):
    jobs = []

    for method in payload['config']['methods']:
        for clustering in payload['config']['clusterings']:
            for encMethod in payload['config']['encodings']:
                encoding = payload['config']['encoding']
                if encoding['generation_type'] == UP_TO:
                    for i in range(1, encoding['prefix_length'] + 1):
                        item = Job.objects.create(
                            status=CREATED,
                            type=generation_type,

                            split=split,
                            encoding=Encoding.objects.get_or_create(
                                data_encoding='label_encoder',
                                # TODO: @HitLuca [value_encoding=,]
                                additional_features=payload['config']['label']['add_remaining_time'] or payload['config']['label'][
                                    'add_elapsed_time'] or
                                                    payload['config']['label']['add_executed_events'] or payload['config']['label'][
                                                        'add_resources_used'] or
                                                    payload['config']['label']['add_new_traces'],
                                temporal_features=payload['config']['label']['add_remaining_time'] or payload['config']['label'][
                                    'add_elapsed_time'],
                                intercase_features=payload['config']['label']['add_executed_events'] or payload['config']['label'][
                                    'add_resources_used'] or
                                                   payload['config']['label']['add_new_traces'],
                                prefix_len=payload['config']['encoding']['prefix_length'],
                                padding=payload['config']['encoding']['padding']
                            ),
                            labelling=Labelling.objects.get_or_create(
                                type=payload['config']['label']['type'],
                                attribute_name=payload['config']['label']['attribute_name'],
                                threshold_type=payload['config']['label']['threshold_type'],
                                threshold=payload['config']['label']['threshold']
                            ),
                            clustering=Clustering.init(clustering, configuration=None),
                            predictive_model=PredictiveModel.init(prediction, configuration=payload)

                        )
                            # config=deepcopy(create_config(payload, encMethod, clustering, method, i)))
                        jobs.append(item)
                else:
                    item = Job.objects.create(
                        status=CREATED,
                        type=generation_type,

                        split=split,
                        encoding=Encoding.objects.get_or_create(
                            data_encoding='label_encoder',
                            # TODO: @HitLuca [value_encoding=,]
                            additional_features=payload['config']['label']['add_remaining_time'] or payload['config']['label'][
                                'add_elapsed_time'] or
                                                payload['config']['label']['add_executed_events'] or payload['config']['label'][
                                                    'add_resources_used'] or
                                                payload['config']['label']['add_new_traces'],
                            temporal_features=payload['config']['label']['add_remaining_time'] or payload['config']['label'][
                                'add_elapsed_time'],
                            intercase_features=payload['config']['label']['add_executed_events'] or payload['config']['label'][
                                'add_resources_used'] or
                                               payload['config']['label']['add_new_traces'],
                            prefix_len=payload['config']['encoding']['prefix_length'],
                            padding=payload['config']['encoding']['padding']
                        ),
                        labelling=Labelling.objects.get_or_create(
                            type=payload['config']['label']['type'],
                            attribute_name=payload['config']['label']['attribute_name'],
                            threshold_type=payload['config']['label']['threshold_type'],
                            threshold=payload['config']['label']['threshold']
                        )
                    )
                        # config=create_config(payload, encMethod, clustering, method, encoding['prefix_length']))
                    jobs.append(item)

    return jobs


def generate_labelling(split, payload):
    jobs = []
    encoding = payload['config']['encoding']
    if encoding['generation_type'] == UP_TO:
        for i in range(1, encoding['prefix_length'] + 1):
            item = Job.objects.create(
                status=CREATED,
                type=LABELLING,

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
        item = Job.objects.create(
            status=CREATED,
            type=LABELLING,

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
                        item = Job.objects.create(
                            split=split,
                            status=CREATED,
                            type=payload['type'],
                            config=deepcopy(create_config(payload, encMethod, clustering, method, i)))
                        jobs.append(item)
                else:
                    item = Job.objects.create(
                        split=split,
                        status=CREATED,
                        type=payload['type'],
                        config=create_config(payload, encMethod, clustering, method, encoding['prefix_length']))
                    jobs.append(item)
    return jobs


def create_config(payload: dict, enc_method: str, clustering: str, method: str, prefix_length: int):
    """Turn lists to single values"""
    config = dict(payload['config'])
    del config['encodings']
    del config['clusterings']
    del config['methods']

    # Extract and merge configurations
    method_conf_name = "{}.{}".format(payload['type'], method)
    method_conf = {**CONF_MAP[method_conf_name](), **payload['config'].get(method_conf_name, dict())}
    # Remove configs that are not needed for this method
    for any_conf_name in ALL_CONFIGS:
        try:
            del config[any_conf_name]
        except KeyError:
            pass
    if clustering == KMEANS:
        config['kmeans'] = {**clustering_kmeans(), **payload['config'].get('kmeans', dict())}
    elif 'kmeans' in config:
        del config['kmeans']
    config[method_conf_name] = method_conf
    config['clustering'] = clustering
    config['method'] = method
    # Encoding stuff rewrite
    config['encoding']['method'] = enc_method
    config['encoding']['prefix_length'] = prefix_length
    return config
