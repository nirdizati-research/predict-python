import shap

from src.encoding.common import retrieve_proper_encoder
from src.explanation.models import Explanation
from sklearn.externals import joblib
import os

def explain(shap_exp: Explanation, training_df, test_df, explanation_target):
    job = shap_exp.job
    model = joblib.load(job.predictive_model.model_path)
    model = model[0]
    shap.initjs()

    training_df = training_df.drop(['trace_id', 'label'], 1)
    test_df = test_df.drop(['trace_id', 'label'], 1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(training_df)

    encoder = retrieve_proper_encoder(job)
    encoder.decode(training_df, job.encoding)
    explanation_target_int = int(explanation_target)

    shap.force_plot(explainer.expected_value[0], shap_values[0][explanation_target_int, :], training_df.iloc[explanation_target_int, :],
                                   show=False, matplotlib=True).savefig("temporal_shap.svg")
    f = open("temporal_shap.svg", "r")
    response = f.read()
    os.remove("temporal_shap.svg")
    return response
