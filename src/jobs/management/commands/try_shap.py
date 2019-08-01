import shap
from django.core.management.base import BaseCommand
from sklearn.externals import joblib

from src.core.core import get_encoded_logs
from src.jobs.models import Job


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):

        #get model
        TARGET_MODEL=1090
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)
        model = model[0]

        #load data
        training_df, test_df = get_encoded_logs(job)

        #get radom point in evaluation set
        EXPLANATION_TARGET = 1

        #get the actual explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(training_df)

        #show plot
        shap.summary_plot(shap_values, training_df)
        shap.summary_plot(shap_values, training_df, plot_type="bar")

        #TODO not yet working
        shap.force_plot(explainer.expected_value, shap_values[EXPLANATION_TARGET, :], training_df.iloc[EXPLANATION_TARGET, :])
        shap.force_plot(explainer.expected_value, shap_values, training_df)
        shap.dependence_plot("RM", shap_values, training_df)
        shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], test_df.iloc[0, :], link="logit") #TODO subst with EXPLANATION_TARGET
        shap.force_plot(explainer.expected_value[0], shap_values[0], test_df, link="logit")

        print('done')
