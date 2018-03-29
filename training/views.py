import json
import os

import django_rq
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.generics import ListAPIView, GenericAPIView
from rest_framework.mixins import RetrieveModelMixin
from rest_framework.response import Response

from training.models import PredModels
from .serializers import ModelSerializer


@api_view(['GET'])
def ModelList(request):
    
    models=PredModels.objects.all()
    """type = request.query_params.get('type', None)
    if type is not None:
        models = models.filter(type=type)"""
    serializer = ModelSerializer(models, many=True)
    return Response(serializer.data, status=201)
    