import time

from src.clustering.models import Clustering, ClusteringMethods
from src.encoding.encoding_container import UP_TO
from src.encoding.models import Encoding, ValueEncodings
from src.hyperparameter_optimization.models import HyperparameterOptimization, HyperparameterOptimizationMethods
from src.jobs.models import Job, JobStatuses, JobTypes
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel
from src.predictive_model.models import PredictiveModels
from src.utils.django_orm import duplicate_orm_row


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
                            task_generation_type=config['encoding'].get('generation_type', 'only_this'),
                            features=config['encoding'].get('features', [])
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
                            clustering=Clustering.init(clustering, configuration=config.get(clustering, {}))
                            if predictive_model.predictive_model != PredictiveModels.TIME_SERIES_PREDICTION.value
                            else Clustering.init(ClusteringMethods.NO_CLUSTER.value, configuration={}),
                            # TODO TEMPORARY workaround,
                            hyperparameter_optimizer=HyperparameterOptimization.init(
                                config.get('hyperparameter_optimizer', {
                                    'type': None}) if predictive_model.predictive_model != PredictiveModels.TIME_SERIES_PREDICTION.value else {
                                    'type': None}),
                            # TODO TEMPORARY workaround
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
                            task_generation_type=config['encoding'].get('generation_type', 'only_this'),
                            features=config['encoding'].get('features', [])
                        )[0],
                        labelling=Labelling.objects.get_or_create(
                            type=labelling_config.get('type', None),
                            # TODO static check?
                            attribute_name=labelling_config.get('attribute_name', None),
                            threshold_type=labelling_config.get('threshold_type', None),
                            threshold=labelling_config.get('threshold', None)
                        )[0] if labelling_config != {} else None,
                        clustering=Clustering.init(clustering, configuration=config.get(clustering, {}))
                        if predictive_model.predictive_model != PredictiveModels.TIME_SERIES_PREDICTION.value
                        else Clustering.init(ClusteringMethods.NO_CLUSTER.value, configuration={}),
                        hyperparameter_optimizer=HyperparameterOptimization.init(
                            config.get('hyperparameter_optimizer', {
                                'type': 'none'}) if predictive_model.predictive_model != PredictiveModels.TIME_SERIES_PREDICTION.value else {
                                'type': 'none'}),
                        # TODO TEMPORARY workaround
                        predictive_model=predictive_model,
                        create_models=config.get('create_models', False)
                    )[0]

                    check_predictive_model_not_overwrite(job)
                    set_model_name(job)

                    jobs.append(job)

    return jobs


def check_predictive_model_not_overwrite(job: Job) -> None:
    if job.hyperparameter_optimizer.optimization_method != HyperparameterOptimizationMethods.NONE.value:
        job.predictive_model = duplicate_orm_row(PredictiveModel.objects.filter(pk=job.predictive_model.pk)[0])
        job.predictive_model.save()
        job.save()


def get_prediction_method_config(predictive_model, prediction_method, payload):
    return {
        'predictive_model': predictive_model,
        'prediction_method': prediction_method,
        **payload.get(predictive_model + '.' + prediction_method, {})
    }


def set_model_name(job: Job) -> None:
    if job.create_models:
        if job.predictive_model.model_path != '':
            job.predictive_model = duplicate_orm_row(PredictiveModel.objects.filter(pk=job.predictive_model.pk)[0])
            job.predictive_model.save()
            job.save()

        if job.clustering.clustering_method != ClusteringMethods.NO_CLUSTER.value:
            job.clustering.model_path = 'cache/model_cache/job_{}-split_{}-clusterer-{}-v0.sav'.format(
                job.id,
                job.split.id,
                job.type)
            job.clustering.save()

        if job.type == JobTypes.UPDATE.value:
            job.type = JobTypes.PREDICTION.value #TODO: Y am I doing this?
            predictive_model_filename = 'cache/model_cache/job_{}-split_{}-predictive_model-{}-v{}.sav'.format(
                job.id,
                job.split.id,
                job.type,
                str(time.time()))
        else:
            predictive_model_filename = 'cache/model_cache/job_{}-split_{}-predictive_model-{}-v0.sav'.format(
                job.id,
                job.split.id,
                job.type)
        job.predictive_model.model_path = predictive_model_filename
        job.predictive_model.save()
        job.save()


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
                    task_generation_type=config['encoding'].get('generation_type', 'only_this'),
                    features=config['encoding'].get('features', [])
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
                task_generation_type=config['encoding'].get('generation_type', 'only_this'),
                features=config['encoding'].get('features', [])
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
                            type=JobTypes.UPDATE.value,
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
                                task_generation_type=config['encoding'].get('generation_type', 'only_this'),
                                features=config['encoding'].get('features', [])
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
                            hyperparameter_optimizer=HyperparameterOptimization.init(
                                config.get('hyperparameter_optimizer', None)),
                            create_models=config.get('create_models', False),
                            incremental_train=Job.objects.filter(
                                pk=config['incremental_train'].get('base_model', None)
                            )[0]
                        )
                        jobs.append(item)
                else:
                    item, _ = Job.objects.get_or_create(
                        status=JobStatuses.CREATED.value,
                        type=JobTypes.UPDATE.value,

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
                            task_generation_type=config['encoding'].get('generation_type', 'only_this'),
                            features=config['encoding'].get('features', [])
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
                        hyperparameter_optimizer=HyperparameterOptimization.init(
                            config.get('hyperparameter_optimizer', None)),
                        create_models=config.get('create_models', False),
                        incremental_train=Job.objects.filter(
                            pk=config['incremental_train'].get('base_model', None)
                        )[0]
                    )
                    jobs.append(item)
    return jobs
