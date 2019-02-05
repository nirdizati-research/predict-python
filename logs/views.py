import logging

from rest_framework import status, mixins, generics
from rest_framework.decorators import api_view
from rest_framework.response import Response

from logs.log_service import create_log
from logs.models import Split
from logs.serializers import SplitSerializer, CreateSplitSerializer
from utils.log_metrics import events_by_date, resources_by_date, event_executions, new_trace_start, trace_attributes, \
    events_in_trace
from .models import Log
from .serializers import LogSerializer

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


@api_view(['GET'])
def get_log_stats(request, pk, stat):
    """Get log statistics

    DEPRECATED ENDPOINT. LOGS HAVE PROPERTIES.

    End URL with
    * events for event_by_date
    * resources for resources_by_date
    * executions for event_executions
    * traceAttributes for trace_attributes
    * eventsInTrace for events_in_trace
    * newTraces for new_trace_start
    """
    try:
        log = Log.objects.get(pk=pk)
    except Log.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)
    try:
        log_file = log.get_file()
    except FileNotFoundError:
        logger.error("Log id: %s, path %s not found", log.id, log.path)
        return Response({'error': 'log file not found'}, status=status.HTTP_404_NOT_FOUND)

    if stat == 'events':
        data = events_by_date(log_file)
    elif stat == 'resources':
        data = resources_by_date(log_file)
    elif stat == 'traceAttributes':
        data = trace_attributes(log_file)
    elif stat == 'eventsInTrace':
        data = events_in_trace(log_file)
    elif stat == 'executions':
        data = event_executions(log_file)
    elif stat == 'newTraces':
        data = new_trace_start(log_file)
    else:
        print('stats error in get_log_stats, setting data to None')
        data = None
    return Response(data)


class SplitList(mixins.ListModelMixin, generics.GenericAPIView):
    queryset = Split.objects.all()
    serializer_class = SplitSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    @staticmethod
    def post(request):
        serializer = CreateSplitSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        # Other serializer for data
        item = serializer.save()
        result = SplitSerializer(item)
        return Response(result.data, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def upload_multiple(request):
    test_log = create_log(request.FILES['testSet'], request.FILES['testSet'].name)
    training_log = create_log(request.FILES['trainingSet'], request.FILES['trainingSet'].name)

    item = Split.objects.create(
        type='double',
        training_log=training_log,
        test_log=test_log)
    serializer = SplitSerializer(item)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


class SplitDetail(mixins.RetrieveModelMixin, generics.GenericAPIView):
    queryset = Split.objects.all()
    serializer_class = SplitSerializer

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)
