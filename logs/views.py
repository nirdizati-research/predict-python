import logging

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from logs.log_service import events_by_date, resources_by_date, event_executions
from .models import Log
from .serializers import LogSerializer

logger = logging.getLogger(__name__)


@api_view(['GET'])
def log_list(request):
    """List of logs with id and name"""
    logs = Log.objects.all()
    serializer = LogSerializer(logs, many=True)
    logger.info("Returned {} logs".format(len(logs)))
    return Response(serializer.data)


@api_view(['GET'])
def get_log_stats(request, pk, stat):
    """Get log statistics

    End URL with
    * events for event_by_date
    * resources for resources_by_date
    * executions for event_executions
    """
    try:
        log = Log.objects.get(pk=pk)
    except Log.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)
    try:
        log_file = log.get_file()
    except FileNotFoundError:
        logger.error("Log id: {}, path {} not found".format(log.id, log.path))
        return Response({'error': 'log file not found'}, status=status.HTTP_404_NOT_FOUND)

    if stat == 'events':
        data = events_by_date(log_file)
    elif stat == 'resources':
        data = resources_by_date(log_file)
    else:
        data = event_executions(log_file)
    return Response(data)
