import json

import django_rq
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.generics import ListAPIView, GenericAPIView
from rest_framework.mixins import RetrieveModelMixin
from rest_framework.response import Response
from .replayer import Replayer

from core.constants import CLASSIFICATION, NEXT_ACTIVITY, REGRESSION
from jobs.serializers import JobSerializer
from logs.models import Split, Log
from training.models import PredModels
from jobs.models import Job, CREATED


@api_view(['GET'])
def demo(request, pk):
    """Get log statistics

    End URL with
    * events for event_by_date
    * resources for resources_by_date
    * executions for event_executions
    """
    """
    config = {'key': 123,
                       'method': 'randomForest',
                       'encoding': 'simpleIndex',
                       'clustering': 'noCluster',
                       'prefix_length':1,
                       "rule": "remaining_time",
                       'threshold': 'default',
                       'log_id':1
                       }
    log = Log.objects.get(pk=3)
    jobrun=Job.objects.create(config=config, type=CLASSIFICATION)
    """
    
    replay=Replayer(pk)
    replay.start()
    #training(jobrun,model)
    #os.system('python3 manage.py rqworker --burst')
    #serializer = JobSerializer(jobrun)
    return Response("Finito")