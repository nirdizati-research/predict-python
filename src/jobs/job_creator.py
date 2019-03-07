from src.clustering.models import Clustering
from src.encoding.encoding_container import UP_TO
from src.encoding.models import Encoding, ValueEncodings
from src.jobs.models import Job, JobStatuses, JobTypes
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel
from src.predictive_model.models import PredictiveModels


def generate(split, payload, generation_type=PredictiveModels.CLASSIFICATION.value):
    jobs = []

    config = payload['config']
    label = config['label'] if 'label' in config else {}

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
                                data_encoding='label_encoder',
                                value_encoding=encMethod,
                                add_elapsed_time=label.get('add_elapsed_time', False),
                                add_remaining_time=label.get('add_remaining_time', False),
                                add_executed_events=label.get('add_executed_events', False),
                                add_resources_used=label.get('add_resources_used', False),
                                add_new_traces=label.get('add_new_traces', False),
                                prefix_length=i,
                                # TODO static check?
                                padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                                task_generation_type=config['encoding'].get('generation_type', 'only_this')
                            )[0],
                            labelling=Labelling.objects.get_or_create(
                                type=label['type'],
                                # TODO static check?
                                attribute_name=
                                label['attribute_name'] if label['attribute_name'] is not None else 'label',
                                threshold_type=label['threshold_type'],
                                threshold=label['threshold']
                            )[0],
                            clustering=Clustering.init(clustering, configuration=None),
                            predictive_model=PredictiveModel.init(
                                get_prediction_method_config(generation_type, method, payload)
                            )
                        )[0]
                        jobs.append(item)
                else:
                    item = Job.objects.get_or_create(
                        status=JobStatuses.CREATED.value,
                        type=generation_type,
                        split=split,
                        encoding=Encoding.objects.get_or_create(
                            data_encoding='label_encoder',
                            value_encoding=encMethod,
                            add_elapsed_time=label.get('add_elapsed_time', False),
                            add_remaining_time=label.get('add_remaining_time', False),
                            add_executed_events=label.get('add_executed_events', False),
                            add_resources_used=label.get('add_resources_used', False),
                            add_new_traces=label.get('add_new_traces', False),
                            prefix_length=config['encoding']['prefix_length'],
                            # TODO static check?
                            padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                            task_generation_type=config['encoding'].get('generation_type', 'only_this')
                        )[0],
                        labelling=Labelling.objects.get_or_create(
                            type=label['type'],
                            # TODO static check?
                            attribute_name=label['attribute_name'] if label['attribute_name'] is not None else 'label',
                            threshold_type=label['threshold_type'],
                            threshold=label['threshold']
                        )[0],
                        clustering=Clustering.init(clustering, configuration=None),
                        predictive_model=PredictiveModel.init(
                            get_prediction_method_config(generation_type, method, payload)
                        )
                    )[0]
                    jobs.append(item)

    return jobs


def get_prediction_method_config(prediction_type, method, payload):
    if prediction_type == PredictiveModels.CLASSIFICATION.value:
        return {
            'predictive_model': prediction_type,
            'prediction_method': method,
            **payload.get('classification', {})
        }
    elif prediction_type == PredictiveModels.REGRESSION.value:
        return {
            'predictive_model': prediction_type,
            'prediction_method': method,
            **payload.get('regression', {})
        }
    elif prediction_type == PredictiveModels.TIME_SERIES_PREDICTION.value:
        return {
            'predictive_model': prediction_type,
            'prediction_method': method,
            **payload.get('time_series_prediction', {})
        }
    else:
        raise ValueError('prediction_type ', prediction_type, 'not recognized')


def generate_labelling(split, payload):
    jobs = []

    encoding = payload['config']['encoding']
    config = payload['config']
    label = config['label'] if 'label' in config else {}

    if encoding['generation_type'] == UP_TO:
        for i in range(1, encoding['prefix_length'] + 1):
            item = Job.objects.get_or_create(
                status=JobStatuses.CREATED.value,
                type=JobTypes.LABELLING.value,

                split=split,
                encoding=Encoding.objects.get_or_create(  # TODO fixme
                    data_encoding='label_encoder',
                    value_encoding=encoding.get('encodings', ValueEncodings.SIMPLE_INDEX.value),
                    add_elapsed_time=label.get('add_elapsed_time', False),
                    add_remaining_time=label.get('add_remaining_time', False),
                    add_executed_events=label.get('add_executed_events', False),
                    add_resources_used=label.get('add_resources_used', False),
                    add_new_traces=label.get('add_new_traces', False),
                    prefix_length=i,
                    # TODO static check?
                    padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                    task_generation_type=config['encoding'].get('generation_type', 'only_this')
                )[0],
                labelling=Labelling.objects.get_or_create(
                    type=label['type'],
                    # TODO static check?
                    attribute_name=label['attribute_name'] if label['attribute_name'] is not None else 'label',
                    threshold_type=label['threshold_type'],
                    threshold=label['threshold']
                )[0]
            )
            jobs.append(item)
    else:
        item = Job.objects.get_or_create(
            status=JobStatuses.CREATED.value,
            type=JobTypes.LABELLING.value,

            split=split,
            encoding=Encoding.objects.get_or_create(  # TODO fixme
                data_encoding='label_encoder',
                value_encoding=encoding.get('encodings', ValueEncodings.SIMPLE_INDEX.value),
                add_elapsed_time=label.get('add_elapsed_time', False),
                add_remaining_time=label.get('add_remaining_time', False),
                add_executed_events=label.get('add_executed_events', False),
                add_resources_used=label.get('add_resources_used', False),
                add_new_traces=label.get('add_new_traces', False),
                prefix_length=config['encoding']['prefix_length'],
                # TODO static check?
                padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                task_generation_type=config['encoding'].get('generation_type', 'only_this')
            )[0],
            labelling=Labelling.objects.get_or_create(
                type=label['type'],
                # TODO static check?
                attribute_name=label['attribute_name'] if label['attribute_name'] is not None else 'label',
                threshold_type=label['threshold_type'],
                threshold=label['threshold']
            )[0]
        )
        jobs.append(item)

    return jobs


def update(split, payload, generation_type=PredictiveModels.CLASSIFICATION.value):  # TODO adapt to allow selecting the predictive_model to update
    jobs = []

    config = payload['config']
    label = config['label'] if 'label' in config else {}

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
                            encoding=Encoding.objects.get_or_create(  # TODO fixme
                                data_encoding='label_encoder',
                                value_encoding=encMethod,
                                add_elapsed_time=label.get('add_elapsed_time', False),
                                add_remaining_time=label.get('add_remaining_time', False),
                                add_executed_events=label.get('add_executed_events', False),
                                add_resources_used=label.get('add_resources_used', False),
                                add_new_traces=label.get('add_new_traces', False),
                                prefix_length=i,
                                # TODO static check?
                                padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                                task_generation_type=config['encoding'].get('generation_type', 'only_this')
                            ),
                            labelling=Labelling.objects.get_or_create(
                                type=label['type'],
                                # TODO static check?
                                attribute_name=
                                label['attribute_name'] if label['attribute_name'] is not None else 'label',
                                threshold_type=label['threshold_type'],
                                threshold=label['threshold']
                            ),
                            clustering=Clustering.init(clustering, configuration=None),
                            predictive_model=PredictiveModel.init(
                                get_prediction_method_config(generation_type, method, payload)
                            )
                        )
                        jobs.append(item)
                else:
                    item = Job.objects.get_or_create(
                        status=JobStatuses.CREATED.value,
                        type=payload['type'],

                        split=split,
                        encoding=Encoding.objects.get_or_create(  # TODO fixme
                            data_encoding='label_encoder',
                            value_encoding=encMethod,
                            add_elapsed_time=label.get('add_elapsed_time', False),
                            add_remaining_time=label.get('add_remaining_time', False),
                            add_executed_events=label.get('add_executed_events', False),
                            add_resources_used=label.get('add_resources_used', False),
                            add_new_traces=label.get('add_new_traces', False),
                            prefix_length=config['encoding']['prefix_length'],
                            # TODO static check?
                            padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                            task_generation_type=config['encoding'].get('generation_type', 'only_this')
                        ),
                        labelling=Labelling.objects.get_or_create(
                            type=label['type'],
                            # TODO static check?
                            attribute_name=label['attribute_name'] if label['attribute_name'] is not None else 'label',
                            threshold_type=label['threshold_type'],
                            threshold=label['threshold']
                        ),
                        clustering=Clustering.init(clustering, configuration=None),
                        predictive_model=PredictiveModel.init(
                            get_prediction_method_config(generation_type, method, payload)
                        )
                    )
                    jobs.append(item)
    return jobs
