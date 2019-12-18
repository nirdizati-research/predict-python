
import lime
import pandas as pd
from sklearn.externals import joblib

from src.encoding.common import retrieve_proper_encoder
from src.explanation.models import Explanation


def explain(lime_exp: Explanation, training_df, test_df, explanation_target=1):
    model = joblib.load(lime_exp.predictive_model.model_path)
    if len(model) > 1:
        raise NotImplementedError('Models with cluster-based approach are not yet supported')
    # get the actual explanation
    features = list(training_df.drop(['trace_id', 'label'], 1).columns.values)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_df.drop(['trace_id', 'label'], 1).as_matrix(),
        feature_names=features,
        categorical_features=[i for i in range(len(list(training_df.drop(['trace_id', 'label'], 1).columns.values)))],
        verbose=True,
        mode='classification',
    )
    explanation_target_vector = test_df.drop(['trace_id', 'label'], 1).iloc[explanation_target]
    exp = explainer.explain_instance(
        explanation_target_vector,
        # TODO probably the opposite would be way less computationally intesive
        model[0].predict_proba,  # TODO if we have clustering this is using only first model
        num_features=len(features)
    )

    # show plot
    #exp.show_in_notebook(show_table=True)
    #exp.as_pyplot_figure().show()
    # exp.save_to_file('/tmp/oi.html')

    # alternative visualisation
    # exp.as_map()

    encoder = retrieve_proper_encoder(lime_exp.job)

    exp_list = exp.as_list()

    explanation_target_df = explanation_target_vector.to_frame().T
    encoder.decode(df=explanation_target_df, encoding=lime_exp.job.encoding)

    exp_list_1 = [(feat, str(explanation_target_df[feat][explanation_target])) for feat in explanation_target_df]
    exp_list = [(exp_list_1[index], exp_list[index][1]) for index in range(len(exp_list))]
    return exp_list
