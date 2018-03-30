import logging

from rest_framework import status, mixins, generics
from rest_framework.decorators import api_view
from rest_framework.response import Response

from logs.log_service import events_by_date, resources_by_date, event_executions, trace_attributes
from logs.models import Split
from logs.serializers import SplitSerializer, CreateSplitSerializer
from .models import Log
from .serializers import LogSerializer

logger = logging.getLogger(__name__)


class LogList(mixins.ListModelMixin, generics.GenericAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request):
        name = self.request.FILES['single'].name
        path = 'log_cache/' + name
        save_file(self.request.FILES['single'], path)
        log = Log.objects.create(name=name, path=path)
        serializer = LogSerializer(log)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

class LogListRun(mixins.ListModelMixin, generics.GenericAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request):
        name = self.request.FILES['single'].name
        path = 'log_run_cache/' + name
        save_file(self.request.FILES['single'], path)
        log = Log.objects.create(name=name, path=path)
        serializer = LogSerializer(log)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def get_log_stats(request, pk, stat):
    """Get log statistics

    End URL with
    * events for event_by_date
    * resources for resources_by_date
    * executions for event_executions
    * traceAttributes for trace_attributes
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
    else:
        data = event_executions(log_file)
    return Response(data)


def save_file(file, path):
    logger.info("Saving uploaded file to %s ", path)
    with open(path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)


class SplitList(mixins.ListModelMixin, generics.GenericAPIView):
    queryset = Split.objects.all()
    serializer_class = SplitSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request):
        serializer = CreateSplitSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        # Other serlializer for data
        item = serializer.save()
        result = SplitSerializer(item)
        return Response(result.data, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def upload_multiple(request):
    test_name = request.FILES['testSet'].name
    training_name = request.FILES['trainingSet'].name
    test_path = 'log_cache/' + test_name
    training_path = 'log_cache/' + training_name
    save_file(request.FILES['testSet'], test_path)
    save_file(request.FILES['trainingSet'], training_path)
    test_log = Log.objects.create(name=test_name, path=test_path)
    training_log = Log.objects.create(name=training_name, path=training_path)

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
