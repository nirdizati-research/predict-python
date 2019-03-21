from src.clustering.models import Clustering
from src.encoding.encoding_container import UP_TO
from src.encoding.models import Encoding, ValueEncodings
from src.jobs.models import Job, JobStatuses, JobTypes
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel
from src.predictive_model.models import PredictiveModels


def generate(split, payload):
    jobs = []

    config = payload['config']
    label = config['label'] if 'label' in config else {}
    job_type = JobTypes.PREDICTION.value
    prediction_type = payload['type']

    for method in config['methods']:
        for clustering in config['clusterings']:
            for encMethod in config['encodings']:
                encoding = config['encoding']
                if encoding['generation_type'] == UP_TO:
                    for i in range(1, encoding['prefix_length'] + 1):
                        item, _ = Job.objects.get_or_create(
                            status=JobStatuses.CREATED.value,
                            type=job_type,
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
                                type=label.get('type', None),
                                # TODO static check?
                                attribute_name=label.get('attribute_name', None),
                                threshold_type=label.get('threshold_type', None),
                                threshold=label.get('threshold', None)
                            )[0] if label != {} else None,
                            clustering=Clustering.init(clustering, configuration=config.get(clustering, {})),
                            predictive_model=PredictiveModel.init(
                                get_prediction_method_config(prediction_type, method, payload)
                            )
                        )
                        jobs.append(item)
                else:
                    item, _ = Job.objects.get_or_create(
                        status=JobStatuses.CREATED.value,
                        type=job_type,
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
                            type=label.get('type', None),
                            # TODO static check?
                            attribute_name=label.get('attribute_name', None),
                            threshold_type=label.get('threshold_type', None),
                            threshold=label.get('threshold', None)
                        )[0] if label != {} else None,
                        clustering=Clustering.init(clustering, configuration=config.get(clustering, {})),
                        predictive_model=PredictiveModel.init(
                            get_prediction_method_config(prediction_type, method, payload)
                        )
                    )
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
        raise ValueError('prediction_type {} not recognized'.format(prediction_type))


def generate_labelling(split, payload):
    jobs = []

    encoding = payload['config']['encoding']
    config = payload['config']
    label = config['label'] if 'label' in config else {}

    if encoding['generation_type'] == UP_TO:
        for i in range(1, encoding['prefix_length'] + 1):
            item, _ = Job.objects.get_or_create(
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
                    type=label.get('type', None),
                    # TODO static check?
                    attribute_name=label.get('attribute_name', None),
                    threshold_type=label.get('threshold_type', None),
                    threshold=label.get('threshold', None)
                )[0] if label != {} else None
            )
            jobs.append(item)
    else:
        item, _ = Job.objects.get_or_create(
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
                type=label.get('type', None),
                # TODO static check?
                attribute_name=label.get('attribute_name', None),
                threshold_type=label.get('threshold_type', None),
                threshold=label.get('threshold', None)
            )[0] if label != {} else None
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
                        item, _ = Job.objects.get_or_create(
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
                            )[0],
                            labelling=Labelling.objects.get_or_create(
                                type=label.get('type', None),
                                # TODO static check?
                                attribute_name=label.get('attribute_name', None),
                                threshold_type=label.get('threshold_type', None),
                                threshold=label.get('threshold', None)
                            )[0] if label != {} else None,
                            clustering=Clustering.init(clustering, configuration=config.get(clustering, {})),
                            predictive_model=PredictiveModel.init(
                                get_prediction_method_config(generation_type, method, payload)
                            )
                        )
                        jobs.append(item)
                else:
                    item, _ = Job.objects.get_or_create(
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
                        )[0],
                        labelling=Labelling.objects.get_or_create(
                            type=label.get('type', None),
                            # TODO static check?
                            attribute_name=label.get('attribute_name', None),
                            threshold_type=label.get('threshold_type', None),
                            threshold=label.get('threshold', None)
                        )[0] if label != {} else None,
                        clustering=Clustering.init(clustering, configuration=config.get(clustering, {})),
                        predictive_model=PredictiveModel.init(
                            get_prediction_method_config(generation_type, method, payload)
                        )
                    )
                    jobs.append(item)
    return jobs
