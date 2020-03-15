from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from src.explanation.explanation import explanation, EXPLAIN, TEMPORAL_STABILITY, explanation_temporal_stability
from src.explanation.models import Explanation, ExplanationTypes
from src.jobs.models import Job


@api_view(['GET'])
def get_lime(request, pk, explanation_target):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.LIME.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()

    error, result = explanation(exp.id, explanation_target)

    if error == 'True':
        return Response({'error': 'Explanation Target cannot be greater than ' + str(result)},
                        status=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
    else:
        return Response(result, status=200)


@api_view(['GET'])
def get_lime_temporal_stability(request, pk, explanation_target=None):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.LIME.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()

    error, result = explanation_temporal_stability(exp.id, explanation_target=explanation_target)

    if error == 'True':
        return Response({'error': 'Explanation Target cannot be greater than ' + str(result)},
                        status=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
    else:
        return Response(result, status=200)


@api_view(['GET'])
def get_temporal_stability(request, pk, explanation_target=None):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.TEMPORAL_STABILITY.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()

    error, result = explanation_temporal_stability(exp.id, explanation_target=explanation_target)

    if error == 'True':
        return Response({'error': 'Explanation Target cannot be greater than ' + str(result)},
                        status=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
    else:
        return Response(result, status=200)


@api_view(['GET'])
def get_shap(request, pk, explanation_target):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.SHAP.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()
    result = explanation(exp.id, explanation_target)
    return Response(result, status=200)


@api_view(['GET'])
def get_anchor(request, pk):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.ANCHOR.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()
    result = explanation(exp.id, explanation_target=None)
    return Response(result, status=200)
