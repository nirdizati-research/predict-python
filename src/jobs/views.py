import json

import django_rq
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.generics import ListAPIView, GenericAPIView
from rest_framework.mixins import RetrieveModelMixin
from rest_framework.response import Response

from src.clustering.models import Clustering
from src.encoding.common import retrieve_proper_encoder, get_encoded_logs
from src.encoding.models import Encoding
from src.hyperparameter_optimization.models import HyperparameterOptimization
from src.jobs import tasks
from src.jobs.job_creator import generate, generate_labelling, update
from src.jobs.models import Job, JobTypes
from src.jobs.serializers import JobSerializer
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel, PredictiveModels
from src.split.models import Split


class JobList(ListAPIView):
    """
    List all jobs, or create a new job.
    """
    serializer_class = JobSerializer

    def get_queryset(self):
        jobs = Job.objects.all()

        type = self.request.data.get('type', None)
        status = self.request.data.get('status', None)
        create_models = self.request.data.get('create_models', None)
        split = self.request.data.get('split', None)

        encoding_config = self.request.data.get('encoding', None)
        labelling_config = self.request.data.get('labelling', None)
        clustering_config = self.request.data.get('clustering', None)
        predictive_model_config = self.request.data.get('predictive_model', None)
        hyperparameter_optimization_config = self.request.data.get('hyperparameter_optimization', None)
        # incremental_train_config = self.request.data.get('incremental_train', None)

        if type is not None:
            jobs = jobs.filter(type=type)
        if status is not None:
            jobs = jobs.filter(status=status)
        if create_models is not None:
            jobs = jobs.filter(create_models=create_models)
        if split is not None:
            jobs = jobs.filter(split=split)
        if encoding_config is not None:
            encodings = Encoding.objects.all()
            if 'data_encoding' in encoding_config:
                encodings = encodings.filter(data_encoding=encoding_config['data_encoding'])
            if 'value_encoding' in encoding_config:
                encodings = encodings.filter(value_encoding=encoding_config['value_encoding'])
            if 'add_elapsed_time' in encoding_config:
                encodings = encodings.filter(add_elapsed_time=encoding_config['add_elapsed_time'])
            if 'add_remaining_time' in encoding_config:
                encodings = encodings.filter(add_remaining_time=encoding_config['add_remaining_time'])
            if 'add_executed_events' in encoding_config:
                encodings = encodings.filter(add_executed_events=encoding_config['add_executed_events'])
            if 'add_resources_used' in encoding_config:
                encodings = encodings.filter(add_resources_used=encoding_config['add_resources_used'])
            if 'add_new_traces' in encoding_config:
                encodings = encodings.filter(add_new_traces=encoding_config['add_new_traces'])
            if 'features' in encoding_config:
                encodings = encodings.filter(features=encoding_config['features'])
            if 'prefix_length' in encoding_config:
                encodings = encodings.filter(prefix_length=encoding_config['prefix_length'])
            if 'padding' in encoding_config:
                encodings = encodings.filter(padding=encoding_config['padding'])
            if 'task_generation_type' in encoding_config:
                encodings = encodings.filter(task_generation_type=encoding_config['task_generation_type'])
            jobs = jobs.filter(encoding__in=[element.id for element in encodings])
        if labelling_config is not None:
            labellings = Labelling.objects.all()
            if 'type' in labelling_config:
                labellings = labellings.filter(type=labelling_config['type'])
            if 'attribute_name' in labelling_config:
                labellings = labellings.filter(attribute_name=labelling_config['attribute_name'])
            if 'threshold_type' in labelling_config:
                labellings = labellings.filter(threshold_type=labelling_config['threshold_type'])
            if 'threshold' in labelling_config:
                labellings = labellings.filter(threshold=labelling_config['threshold'])
            jobs = jobs.filter(labelling__in=[element.id for element in labellings])
        if clustering_config is not None:
            clusterings = Clustering.objects.all()
            if 'clustering_method' in clustering_config:
                clusterings = clusterings.filter(clustering_method=clustering_config['clustering_method'])
            jobs = jobs.filter(clustering__in=[element.id for element in clusterings])
        if predictive_model_config is not None:
            predictive_models = PredictiveModel.objects.all()
            if 'predictive_model' in predictive_model_config:
                predictive_models = predictive_models.filter(predictive_model=predictive_model_config['predictive_model'])
            if 'prediction_method' in predictive_model_config:
                predictive_models = predictive_models.filter(prediction_method=predictive_model_config['prediction_method'])
            jobs = jobs.filter(predictive_model__in=[element.id for element in predictive_models])
        if hyperparameter_optimization_config is not None:
            hyperparameter_optimizations = HyperparameterOptimization.objects.all()
            if 'optimization_method' in hyperparameter_optimization_config:
                hyperparameter_optimizations = hyperparameter_optimizations.filter(optimization_method=hyperparameter_optimization_config['optimization_method'])

            # if 'max_evaluations' in hyperparameter_optimization_config: #TODO add support for inner parameters of hyperopt
            #     hyperparameter_optimizations.filter(max_evaluations=hyperparameter_optimization_config['max_evaluations'])
            # if 'performance_metric' in hyperparameter_optimization_config:
            #     hyperparameter_optimizations.filter(performance_metric=hyperparameter_optimization_config['performance_metric'])
            # if 'algorithm_type' in hyperparameter_optimization_config:
            #     hyperparameter_optimizations.filter(algorithm_type=hyperparameter_optimization_config['algorithm_type'])
            jobs = jobs.filter(hyperparameter_optimizer__in=[element.id for element in hyperparameter_optimizations])
        # elif incremental_train_config is not None:
        #     incremental_train = # TODO ADD RECURSION TO THIS FUNCTION
        #     jobs = jobs.filter(incremental_train=incremental_train)

        return jobs

    # TODO remove?
    @staticmethod
    def post(request):
        serializer = JobSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


class JobDetail(RetrieveModelMixin, GenericAPIView):
    queryset = Job.objects.all()
    serializer_class = JobSerializer

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        job = self.queryset.get(pk=kwargs['pk'])
        job.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['POST'])
def create_multiple(request):
    """No request validation"""
    payload = json.loads(request.body.decode('utf-8'))
    try:
        split = Split.objects.get(pk=payload['split_id'])
    except Split.DoesNotExist:
        return Response({'error': 'split_id ' + str(payload['split_id']) + ' not in database'}, status=status.HTTP_404_NOT_FOUND)

    # detect either or not a predictive_model to update has been specified otherwise train a new one.
    if 'incremental_train' in payload['config'] and len(payload['config']['incremental_train']) > 0:
        jobs = update(split, payload)
    elif payload['type'] in [e.value for e in PredictiveModels]:
        jobs = generate(split, payload)
    elif payload['type'] == JobTypes.LABELLING.value:
        jobs = generate_labelling(split, payload)
    else:
        return Response({'error': 'type not supported'.format(payload['type'])},
                        status=status.HTTP_422_UNPROCESSABLE_ENTITY)
    for job in jobs:
        # TODO add support for 'depends_on' parameter
        django_rq.enqueue(tasks.prediction_task, job.id)
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)


@api_view(['GET'])
def get_decoded_df(request, pk):
    job = Job.objects.filter(pk=pk)[0]
    training_df, test_df = get_encoded_logs(job)
    training_df = training_df[:100]
    training_df = training_df.drop(['trace_id'], 1)
    encoder = retrieve_proper_encoder(job)
    encoder.decode(training_df, job.encoding)
    return Response(training_df, status=200)


@api_view(['GET'])
def get_unique_values(request, pk):
    job = Job.objects.filter(pk=pk)[0]
    training_df, test_df = get_encoded_logs(job)
    decoded_training_df = training_df.copy()
    decoded_testing_df = test_df.copy()
    training_df = training_df.drop(['trace_id','label'], 1)

    encoder = retrieve_proper_encoder(job)
    encoder.decode(df=decoded_training_df, encoding=job.encoding)
    encoder.decode(df=decoded_testing_df, encoding=job.encoding)

    result_df = {}
    for key in training_df.keys():
        result_decoded_df = list(set(list(training_df[key]) + list(test_df[key])))
        result_encoded_df= list(set(list(decoded_training_df[key]) + list(decoded_testing_df[key])))

        result_df[key] = {}
        for k in range(len(result_decoded_df)):
            result_df[key][result_encoded_df[k]] = result_decoded_df[k]
    return Response(result_df, status=200)
