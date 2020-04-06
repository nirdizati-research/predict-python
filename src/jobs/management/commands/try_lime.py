import lime
import lime.lime_tabular
import matplotlib as plt
from django.core.management.base import BaseCommand
from sklearn.externals import joblib

from src.core.core import get_encoded_logs
from src.encoding.common import retrieve_proper_encoder
from src.jobs.models import Job


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):

        #get model
        TARGET_MODEL=5
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)

        #load data
        training_df, test_df = get_encoded_logs(job)

        #get radom point in evaluation set
        EXPLANATION_TARGET = 3
        #get the actual explanation
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_df.drop(['trace_id', 'label'], 1).as_matrix(),
            feature_names=list(training_df.drop(['trace_id', 'label'], 1).columns.values),
            categorical_features=[i for i in range(len(list(training_df.drop(['trace_id', 'label'], 1).columns.values)))],
            verbose=True,
            mode='classification',
        )
        exp = explainer.explain_instance(
            test_df.drop(['trace_id', 'label'], 1).iloc[EXPLANATION_TARGET], #TODO probably the opposite would be way less computationally intesive
            model[0].predict_proba,
            num_features=5
        )
        exp.as_list()

        #show plot
        #exp.show_in_notebook(show_table=True)
        # exp.as_pyplot_figure().show()
        exp.save_to_file('oi.html')

        print('done')
