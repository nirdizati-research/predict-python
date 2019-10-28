import shap
from src.explanation.models import Explanation


def explain(shap_exp: Explanation, training_df, test_df, explanation_target):
    explainer = shap.TreeExplainer(shap_exp.predictive_model)
    shap_values = explainer.shap_values(training_df)

    # show plot
    shap.summary_plot(shap_values, training_df)
    shap.summary_plot(shap_values, training_df, plot_type="bar")

    # TODO not yet working
    """
	shap.force_plot(explainer.expected_value, shap_values[EXPLANATION_TARGET, :],
					training_df.iloc[EXPLANATION_TARGET, :])
	shap.force_plot(explainer.expected_value, shap_values, training_df)
	shap.dependence_plot("RM", shap_values, training_df)
	shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], test_df.iloc[0, :],
					link="logit")  # TODO subst with EXPLANATION_TARGET
	shap.force_plot(explainer.expected_value[0], shap_values[0], test_df, link="logit")
	"""
    print('done')
    return shap_values
