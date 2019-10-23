import lime
import lime.lime_tabular
import numpy as np
import shap
from anchor import anchor_tabular

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.externals import joblib

from src.core.core import get_encoded_logs, ModelActions, MODEL
from src.encoding.encoder import Encoder
from src.explanation.explanation import explanation
from src.explanation.models import Explanation, ExplanationTypes
from src.jobs.models import Job
from src.split.splitting import get_train_test_log


@api_view(['GET'])
def get_lime(request, pk):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.LIME.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()
    result = explanation(exp.id)
    return Response(result, status=200)


@api_view(['GET'])
def get_shap(request, pk):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.SHAP.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()
    result = explanation(exp.id)
    return Response(result, status=200)


@api_view(['GET'])
def get_anchor(request, pk):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.ANCHOR.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()
    result = explanation(exp.id)
    return Response(result, status=200)
