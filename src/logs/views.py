import logging

# from pm4py.algo.discovery.alpha import factory as alpha_miner
from rest_framework import status, mixins, generics
from rest_framework.decorators import api_view
from rest_framework.response import Response

from src.logs.log_service import create_log, get_log_trace_attributes
from src.split.models import Split
from src.split.serializers import SplitSerializer
from .models import Log
from .serializers import LogSerializer
from src.utils.file_service import get_log


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
def get_log_traces_attributes(request, pk):
    log = Log.objects.get(pk=pk)
    try:
        log_file = get_log(log)
    except FileNotFoundError:
        logger.error("Log id: %s, path %s not found", log.id, log.path)
        return Response({'error': 'log file not found'}, status=status.HTTP_404_NOT_FOUND)
    value = get_log_trace_attributes(log_file)
    return Response(value, status=200)


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


class SplitDetail(mixins.RetrieveModelMixin, generics.GenericAPIView):
    queryset = Split.objects.all()
    serializer_class = SplitSerializer

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)
