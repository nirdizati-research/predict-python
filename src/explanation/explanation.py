from src.core.core import get_encoded_logs
from src.explanation import lime_wrapper, shap_wrapper, anchor_wrapper, temporal_stability, \
    ice_wrapper, skater_wrapper, cm_feedback_wrapper, retrain_wrapper
from src.explanation.models import Explanation, ExplanationTypes

EXPLAIN = 'explain'
TEMPORAL_STABILITY = 'temporal_stability'

EXPLANATION = {
    ExplanationTypes.LIME.value: {
        'explain': lime_wrapper.explain,
        'temporal_stability': lime_wrapper.lime_temporal_stability
    },
    ExplanationTypes.SHAP.value: {
        'explain': shap_wrapper.explain,
        'temporal_stability': shap_wrapper.shap_temporal_stability
    },
    ExplanationTypes.ICE.value: {
        'explain': ice_wrapper.explain
    },
    ExplanationTypes.SKATER.value: {
        'explain': skater_wrapper.explain
    },
    ExplanationTypes.CMFEEDBACK.value: {
        'explain': cm_feedback_wrapper.explain
    },
    ExplanationTypes.RETRAIN.value: {
        'explain': retrain_wrapper.explain
    },
    ExplanationTypes.ANCHOR.value: {
        'explain': anchor_wrapper.explain
    },
    ExplanationTypes.TEMPORAL_STABILITY.value:{
        'temporal_stability': temporal_stability.temporal_stability
    }
}


def explanation(exp_id: int, explanation_target: str = None, prefix_target: str = None):
    exp = Explanation.objects.filter(pk=exp_id)[0]
    job = exp.job
    # load data
    training_df, test_df = get_encoded_logs(job)

    try:
        result = EXPLANATION[exp.type][EXPLAIN](exp, training_df, test_df, explanation_target, prefix_target)
        return 'False', result
    except Exception as e:
        return 'True', str(e)


def explanation_temporal_stability(exp_id: int, explanation_target: str = None):
    exp = Explanation.objects.filter(pk=exp_id)[0]
    job = exp.job
    # load data
    training_df, test_df = get_encoded_logs(job)
    try:
        result = EXPLANATION[exp.type][TEMPORAL_STABILITY](exp, training_df, test_df, explanation_target)
        return 'False', result
    except Exception as e:
        return 'True', str(e)
