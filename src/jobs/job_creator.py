from src.clustering.models import Clustering
from src.encoding.encoding_container import UP_TO
from src.encoding.models import Encoding, ValueEncodings
from src.hyperparameter_optimization.models import HyperparameterOptimization
from src.jobs.models import Job, JobStatuses, JobTypes
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel
from src.predictive_model.models import PredictiveModels


def generate(split, payload):
    jobs = []

    config = payload['config']
    labelling_config = config['labelling'] if 'labelling' in config else {}
    job_type = JobTypes.PREDICTION.value
    prediction_type = payload['type']

    for method in config['methods']:
        for clustering in config['clusterings']:
            for encMethod in config['encodings']:
                encoding = config['encoding']
                if encoding['generation_type'] == UP_TO:
                    for i in range(1, encoding['prefix_length'] + 1):
                        encoding = Encoding.objects.get_or_create(
                            data_encoding='label_encoder',
                            value_encoding=encMethod,
                            add_elapsed_time=labelling_config.get('add_elapsed_time', False),
                            add_remaining_time=labelling_config.get('add_remaining_time', False),
                            add_executed_events=labelling_config.get('add_executed_events', False),
                            add_resources_used=labelling_config.get('add_resources_used', False),
                            add_new_traces=labelling_config.get('add_new_traces', False),
                            prefix_length=i,
                            # TODO static check?
                            padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                            task_generation_type=config['encoding'].get('generation_type', 'only_this')
                        )[0]

                        predictive_model = PredictiveModel.init(
                            get_prediction_method_config(prediction_type, method, config))

                        job = Job.objects.get_or_create(
                            status=JobStatuses.CREATED.value,
                            type=job_type,
                            split=split,
                            encoding=encoding,
                            labelling=Labelling.objects.get_or_create(
                                type=labelling_config.get('type', None),
                                # TODO static check?
                                attribute_name=labelling_config.get('attribute_name', None),
                                threshold_type=labelling_config.get('threshold_type', None),
                                threshold=labelling_config.get('threshold', None)
                            )[0] if labelling_config != {} else None,
                            clustering=Clustering.init(clustering, configuration=config.get(clustering, {})),
                            hyperparameter_optimizer=HyperparameterOptimization.init(
                                config.get('hyperparameter_optimizer', None)),
                            predictive_model=predictive_model,
                            create_models=config.get('create_models', False)
                        )[0]

                        jobs.append(job)
                else:
                    predictive_model = PredictiveModel.init(
                        get_prediction_method_config(prediction_type, method, config))

                    job = Job.objects.get_or_create(
                        status=JobStatuses.CREATED.value,
                        type=job_type,
                        split=split,
                        encoding=Encoding.objects.get_or_create(
                            data_encoding='label_encoder',
                            value_encoding=encMethod,
                            add_elapsed_time=labelling_config.get('add_elapsed_time', False),
                            add_remaining_time=labelling_config.get('add_remaining_time', False),
                            add_executed_events=labelling_config.get('add_executed_events', False),
                            add_resources_used=labelling_config.get('add_resources_used', False),
                            add_new_traces=labelling_config.get('add_new_traces', False),
                            prefix_length=config['encoding']['prefix_length'],
                            # TODO static check?
                            padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                            task_generation_type=config['encoding'].get('generation_type', 'only_this')
                        )[0],
                        labelling=Labelling.objects.get_or_create(
                            type=labelling_config.get('type', None),
                            # TODO static check?
                            attribute_name=labelling_config.get('attribute_name', None),
                            threshold_type=labelling_config.get('threshold_type', None),
                            threshold=labelling_config.get('threshold', None)
                        )[0] if labelling_config != {} else None,
                        clustering=Clustering.init(clustering, configuration=config.get(clustering, {})),
                        hyperparameter_optimizer=HyperparameterOptimization.init(config.get('hyperparameter_optimizer', None)),
                        predictive_model=predictive_model,
                        create_models=config.get('create_models', False)
                    )[0]
                    jobs.append(job)

    return jobs


def get_prediction_method_config(predictive_model, prediction_method, payload):
    return {
        'predictive_model': predictive_model,
        'prediction_method': prediction_method,
        **payload.get(predictive_model + '.' + prediction_method, {})
    }


def generate_labelling(split, payload):
    jobs = []

    encoding = payload['config']['encoding']
    config = payload['config']
    labelling_config = config['labelling'] if 'labelling' in config else {}

    if encoding['generation_type'] == UP_TO:
        for i in range(1, encoding['prefix_length'] + 1):
            item, _ = Job.objects.get_or_create(
                status=JobStatuses.CREATED.value,
                type=JobTypes.LABELLING.value,

                split=split,
                encoding=Encoding.objects.get_or_create(  # TODO fixme
                    data_encoding='label_encoder',
                    value_encoding=encoding.get('encodings', ValueEncodings.SIMPLE_INDEX.value),
                    add_elapsed_time=labelling_config.get('add_elapsed_time', False),
                    add_remaining_time=labelling_config.get('add_remaining_time', False),
                    add_executed_events=labelling_config.get('add_executed_events', False),
                    add_resources_used=labelling_config.get('add_resources_used', False),
                    add_new_traces=labelling_config.get('add_new_traces', False),
                    prefix_length=i,
                    # TODO static check?
                    padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                    task_generation_type=config['encoding'].get('generation_type', 'only_this')
                )[0],
                labelling=Labelling.objects.get_or_create(
                    type=labelling_config.get('type', None),
                    # TODO static check?
                    attribute_name=labelling_config.get('attribute_name', None),
                    threshold_type=labelling_config.get('threshold_type', None),
                    threshold=labelling_config.get('threshold', None)
                )[0] if labelling_config != {} else None
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
                add_elapsed_time=labelling_config.get('add_elapsed_time', False),
                add_remaining_time=labelling_config.get('add_remaining_time', False),
                add_executed_events=labelling_config.get('add_executed_events', False),
                add_resources_used=labelling_config.get('add_resources_used', False),
                add_new_traces=labelling_config.get('add_new_traces', False),
                prefix_length=config['encoding']['prefix_length'],
                # TODO static check?
                padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                task_generation_type=config['encoding'].get('generation_type', 'only_this')
            )[0],
            labelling=Labelling.objects.get_or_create(
                type=labelling_config.get('type', None),
                # TODO static check?
                attribute_name=labelling_config.get('attribute_name', None),
                threshold_type=labelling_config.get('threshold_type', None),
                threshold=labelling_config.get('threshold', None)
            )[0] if labelling_config != {} else None
        )
        jobs.append(item)

    return jobs


def update(split, payload, generation_type=PredictiveModels.CLASSIFICATION.value):  # TODO adapt to allow selecting the predictive_model to update
    jobs = []

    config = payload['config']
    labelling_config = config['labelling'] if 'labelling' in config else {}

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
                                add_elapsed_time=labelling_config.get('add_elapsed_time', False),
                                add_remaining_time=labelling_config.get('add_remaining_time', False),
                                add_executed_events=labelling_config.get('add_executed_events', False),
                                add_resources_used=labelling_config.get('add_resources_used', False),
                                add_new_traces=labelling_config.get('add_new_traces', False),
                                prefix_length=i,
                                # TODO static check?
                                padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                                task_generation_type=config['encoding'].get('generation_type', 'only_this')
                            )[0],
                            labelling=Labelling.objects.get_or_create(
                                type=labelling_config.get('type', None),
                                # TODO static check?
                                attribute_name=labelling_config.get('attribute_name', None),
                                threshold_type=labelling_config.get('threshold_type', None),
                                threshold=labelling_config.get('threshold', None)
                            )[0] if labelling_config != {} else None,
                            clustering=Clustering.init(clustering, configuration=config.get(clustering, {})),
                            predictive_model=PredictiveModel.init(
                                get_prediction_method_config(generation_type, method, payload)
                            ),
                            create_models=config.get('create_models', False)
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
                            add_elapsed_time=labelling_config.get('add_elapsed_time', False),
                            add_remaining_time=labelling_config.get('add_remaining_time', False),
                            add_executed_events=labelling_config.get('add_executed_events', False),
                            add_resources_used=labelling_config.get('add_resources_used', False),
                            add_new_traces=labelling_config.get('add_new_traces', False),
                            prefix_length=config['encoding']['prefix_length'],
                            # TODO static check?
                            padding=True if config['encoding']['padding'] == 'zero_padding' else False,
                            task_generation_type=config['encoding'].get('generation_type', 'only_this')
                        )[0],
                        labelling=Labelling.objects.get_or_create(
                            type=labelling_config.get('type', None),
                            # TODO static check?
                            attribute_name=labelling_config.get('attribute_name', None),
                            threshold_type=labelling_config.get('threshold_type', None),
                            threshold=labelling_config.get('threshold', None)
                        )[0] if labelling_config != {} else None,
                        clustering=Clustering.init(clustering, configuration=config.get(clustering, {})),
                        predictive_model=PredictiveModel.init(
                            get_prediction_method_config(generation_type, method, payload)
                        ),
                        create_models=config.get('create_models', False)
                    )
                    jobs.append(item)
    return jobs
