import logging

# from pm4py.algo.discovery.alpha import factory as alpha_miner
from django.http import HttpResponse
from rest_framework import status, mixins, generics
from rest_framework.decorators import api_view
from rest_framework.response import Response

from src.logs.log_service import create_log
from src.split.models import Split
from src.split.serializers import SplitSerializer
from .models import Log
from .serializers import LogSerializer
from src.logs.log_service import create_log, get_log_trace_attributes
from ..utils.file_service import get_log
from pm4py.objects.log.importer.xes import factory as xes_importer

logger = logging.getLogger(__name__)


class LogList(mixins.ListModelMixin, generics.GenericAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request):
        log = create_log(self.request.FILES['single'], self.request.FILES['single'].name)
        serializer = LogSerializer(log)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class LogDetail(mixins.RetrieveModelMixin, generics.GenericAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)


# @api_view(['GET'])
# def get_log_stats(request, pk, stat):
#     """Get log statistics
#
#     DEPRECATED ENDPOINT. LOGS HAVE PROPERTIES.
#
#     End URL with
#     * events for event_by_date
#     * resources for resources_by_date
#     * executions for event_executions
#     * traceAttributes for trace_attributes
#     * eventsInTrace for events_in_trace
#     * newTraces for new_trace_start
#     """
#     try:
#         log = Log.objects.get(pk=pk)
#     except Log.DoesNotExist:
#         return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)
#     try:
#         log_file = log.get_file()
#     except FileNotFoundError:
#         logger.error("Log id: %s, path %s not found", log.id, log.path)
#         return Response({'error': 'log file not found'}, status=status.HTTP_404_NOT_FOUND)
#
#     if stat == 'events':
#         data = events_by_date(log_file)
#     elif stat == 'resources':
#         data = resources_by_date(log_file)
#     elif stat == 'traceAttributes':
#         data = trace_attributes(log_file)
#     elif stat == 'eventsInTrace':
#         data = events_in_trace(log_file)
#     elif stat == 'executions':
#         data = event_executions(log_file)
#     elif stat == 'newTraces':
#         data = new_trace_start(log_file)
#     elif stat == 'alpha_miner':
#         data = alpha_miner.apply(log)
#     else:
#         logger.info('stats error in get_log_stats, setting data to None')
#         data = None
#     return Response(data)


@api_view(['GET'])
def get_log_traces_attributes(request, pk):
    log = Log.objects.get(pk=pk)
    try:
        log_file = get_log(log)

    except FileNotFoundError:
        logger.error("Log id: %s, path %s not found", log.id, log.path)
        return Response({'error': 'log file not found'}, status=status.HTTP_404_NOT_FOUND)
    value = get_log_trace_attributes(log_file)
    return Response(value, status=status.HTTP_200_OK)


@api_view(['POST'])
def upload_multiple(request):
    logger.info('Double upload request received.')
    test_log = create_log(request.FILES['testSet'], request.FILES['testSet'].name)
    train_log = create_log(request.FILES['trainingSet'], request.FILES['trainingSet'].name)

    item = Split.objects.create(
        type='double',
        train_log=train_log,
        test_log=test_log)
    serializer = SplitSerializer(item)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def get_split_train_logs(request, pk):
    try:
        value = Split.objects.get(pk=pk)
    except:
        return Response("No file found by the given id", status=status.HTTP_404_NOT_FOUND)
    with open(value.train_log.path, 'r') as f:
        data = f.read()
    response = HttpResponse(data, content_type='application/xes')
    response['Content-Disposition'] = 'attachment; filename=' + value.train_log.name
    return response


@api_view(['GET'])
def get_split_test_logs(request, pk):
    try:
        value = Split.objects.get(pk=pk)
    except:
        return Response("No file found by the given id", status=status.HTTP_404_NOT_FOUND)
    with open(value.test_log.path, 'r') as f:
        data = f.read()
    response = HttpResponse(data, content_type='application/xes')
    response['Content-Disposition'] = 'attachment; filename=' + value.test_log.name
    return response


class SplitDetail(mixins.RetrieveModelMixin, generics.GenericAPIView):
    queryset = Split.objects.all()
    serializer_class = SplitSerializer

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)
