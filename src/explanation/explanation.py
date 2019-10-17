from src.core.core import get_encoded_logs
from src.explanation import lime_wrapper, shap_wrapper, anchor_wrapper
from src.explanation.models import Explanation, ExplanationTypes

EXPLAIN = 'explain'

EXPLANATION = {
    ExplanationTypes.LIME.value: {
        'explain': lime_wrapper.explain
    },
    ExplanationTypes.SHAP.value: {
        'explain': shap_wrapper.explain
    },
    ExplanationTypes.ANCHOR.value: {
        'explain': anchor_wrapper.explain
    }
}


def explanation(exp_id: int):
    exp = Explanation.objects.filter(pk=exp_id)[0]
    job = exp.job
    # load data
    training_df, test_df = get_encoded_logs(job)

    result = EXPLANATION[exp.type][EXPLAIN](exp, training_df, test_df)

    return result

