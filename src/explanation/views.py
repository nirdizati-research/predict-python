from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from src.explanation.explanation import explanation, explanation_temporal_stability
from src.explanation.models import Explanation, ExplanationTypes
from src.jobs.models import Job
import pandas as pd



@api_view(['GET'])
def get_lime(request, pk, explanation_target):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.LIME.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()

    if 'lime' not in exp.results:
        exp.results.update({'lime': dict()})

    if explanation_target in exp.results['lime']:
        return Response(exp.results['lime'][explanation_target], status=200)

    else:
        error, result = explanation(exp.id, explanation_target)

        exp.results['lime'].update({explanation_target: result})
        exp.save()
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

    if 'lime_temporal' not in exp.results:
        exp.results.update({'lime_temporal': dict()})

    if explanation_target:
        if explanation_target in exp.results['lime_temporal']:
            return Response(exp.results['lime_temporal'][explanation_target], status=200)
        else:
            error, result = explanation_temporal_stability(exp.id, explanation_target=explanation_target)

            exp.results['lime_temporal'].update({explanation_target: result})
            exp.save()
            if error == 'True':
                return Response({'error': 'Explanation Target cannot be greater than ' + str(result)},
                                status=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
            else:
                return Response(result, status=200)
    elif 'no_target' in explanation_target:
        return Response(exp.results['lime_temporal']['no_target'], status=200)
    else:
        error, result = explanation_temporal_stability(exp.id, explanation_target=explanation_target)
        exp.results['lime_temporal'].update({'no_target': result})
        exp.save()
        if error == 'True':
            return Response({'error': 'Explanation Target cannot be greater than ' + str(result)},
                            status=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
        else:
            return Response(result, status=200)


@api_view(['GET'])
def get_shap_temporal_stability(request, pk, explanation_target=None):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.SHAP.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()

    if 'shap_temporal' not in exp.results:
        exp.results.update({'shap_temporal': dict()})

    if explanation_target:
        if explanation_target in exp.results['shap_temporal']:
            return Response(exp.results['shap_temporal'][explanation_target], status=200)
        else:
            error, result = explanation_temporal_stability(exp.id, explanation_target=explanation_target)

            #exp.results['shap_temporal'].update({explanation_target: pd.Series(result).to_json(orient='values')})
            #exp.save()
            if error == 'True':
                return Response({'error': 'Explanation Target cannot be greater than ' + str(result)},
                                status=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
            else:
                return Response(result, status=200)
    elif 'no_target' in explanation_target:
        return Response(exp.results['shap_temporal']['no_target'], status=200)
    else:
        error, result = explanation_temporal_stability(exp.id, explanation_target=explanation_target)
        #exp.results['shap_temporal'].update({'no_target': pd.Series(result).to_json(orient='values')})
        #exp.save()
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
    if 'temporal' not in exp.results:
        exp.results.update({'temporal': dict()})

    if explanation_target:
        if explanation_target in exp.results['temporal']:
            return Response(pd.read_json(exp.results['temporal'][explanation_target], typ='series',
                     orient='records'), status = 200)

        else:
            error, result = explanation_temporal_stability(exp.id, explanation_target=explanation_target)

            exp.results['temporal'].update({explanation_target: result})
            exp.save()
            if error == 'True':
                return Response({'error': 'Explanation Target cannot be greater than ' + str(result)},
                                status=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
            else:
                return Response(result, status=200)
    elif 'no_target' in explanation_target:
        return Response(exp.results['temporal']['no_target'], status=200)
    else:
        error, result = explanation_temporal_stability(exp.id, explanation_target=explanation_target)
        exp.results['temporal'].update({'no_target': result})
        exp.save()
        if error == 'True':
            return Response({'error': 'Explanation Target cannot be greater than ' + str(result)},
                            status=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
        else:
            return Response(result, status=200)


@api_view(['GET'])
def get_shap(request, pk, explanation_target, prefix_target):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.SHAP.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)

    exp.save()

    if 'shap' not in exp.results:
        exp.results.update({'shap': dict()})

    if explanation_target not in exp.results['shap']:
        exp.results['shap'] = {explanation_target: dict()}

    if explanation_target in exp.results['shap'] and prefix_target in exp.results['shap'][explanation_target].keys():
        return Response(pd.read_json(exp.results['shap'][explanation_target][prefix_target], typ='series', orient='records'), status=200)

    else:
        result = explanation(exp.id, explanation_target, prefix_target)
        exp.results['shap'][explanation_target].update({prefix_target: pd.Series(result).to_json(orient='values')})
        exp.save()
        return Response(result, status=200)


@api_view(['GET'])
def get_skater(request, pk):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.SKATER.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()
    if 'skater' in exp.results:
        return Response(exp.results['skater'], status=200)
    else:
        result = explanation(exp.id, explanation_target = None)
        exp.results['skater'] = result
        exp.save()
        return Response(result, status=200)


@api_view(['GET'])
def get_ice(request, pk, explanation_target):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.ICE.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()

    if 'ice' not in exp.results:
        exp.results.update({'ice': dict()})

    if explanation_target in exp.results['ice']:
        return Response(exp.results['ice'][explanation_target], status=200)

    else:
        result = explanation(exp.id, explanation_target)

        exp.results['ice'].update({explanation_target: result})
        exp.save()
        return Response(result, status=200)


@api_view(['GET'])
def get_cmfeedback(request, pk, top_k):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.CMFEEDBACK.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()
    result = explanation(exp.id, int(top_k))
    return Response(result, status=200)


@api_view(['POST'])
def get_retrain(request, pk):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.RETRAIN.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()
    target = request.data
    result = explanation(exp.id, target)
    return Response(result, status=200)


@api_view(['GET'])
def get_anchor(request, pk):
    job = Job.objects.filter(pk=pk)[0]
    exp, _ = Explanation.objects.get_or_create(type=ExplanationTypes.ANCHOR.value, split=job.split,
                                               predictive_model=job.predictive_model, job=job)
    exp.save()
    if 'anchor' in exp.results:
        return Response(exp.results['anchor'], status=200)
    else:
        result = explanation(exp.id, explanation_target=None)
        exp.results['anchor'] = result
        exp.save()
        return Response(result, status=200)
