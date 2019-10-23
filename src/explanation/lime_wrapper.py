
import lime
from sklearn.externals import joblib

from src.explanation.models import Explanation


def explain(lime_exp: Explanation, training_df, test_df):
    explanation_target = 1
    model = joblib.load(lime_exp.predictive_model.model_path)
    # get the actual explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_df.drop(['trace_id', 'label'], 1).as_matrix(),
        feature_names=list(training_df.drop(['trace_id', 'label'], 1).columns.values),
        categorical_features=[i for i in range(len(list(training_df.drop(['trace_id', 'label'], 1).columns.values)))],
        verbose=True,
        mode='classification',
    )
    exp = explainer.explain_instance(
        test_df.drop(['trace_id', 'label'], 1).iloc[explanation_target],
        # TODO probably the opposite would be way less computationally intesive
        model[0].predict_proba,
        num_features=5
    )
    exp.as_list()

    # show plot
    #exp.show_in_notebook(show_table=True)
    #exp.as_pyplot_figure().show()
    # exp.save_to_file('/tmp/oi.html')

    return exp.as_map()
