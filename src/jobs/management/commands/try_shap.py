import shap
import sklearn
import xgboost

from django.core.management.base import BaseCommand
from sklearn.externals import joblib

from src.core.core import get_encoded_logs
from src.encoding.common import retrieve_proper_encoder
from src.jobs.models import Job
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import xgboost


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):

        TARGET_MODEL = 53
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)
        model = model[0]
        training_df, test_df = get_encoded_logs(job)

        EXPLANATION_TARGET = 12
        FEATURE_TARGET = 1
        shap.initjs()

        explainer = shap.TreeExplainer(model)
        training_df = training_df.drop(['trace_id','label'], 1)
        test_df = test_df.drop(['trace_id'], 1)

        shap_values = explainer.shap_values(training_df)

        encoder = retrieve_proper_encoder(job)
        encoder.decode(training_df, job.encoding)

        shap.force_plot(explainer.expected_value[1], shap_values[0][EXPLANATION_TARGET, :],training_df.iloc[EXPLANATION_TARGET, :],
                                       show=False, matplotlib=True).savefig('shap_plot_train_1_3.png')

        # encoder.encode(training_df, job.encoding)
        # plt.savefig('dependence_plot.png')
        # a = shap.force_plot(explainer.expected_value[1], shap_values[0], training_df)
        # shap.save_html('shap_plot_train_1_3.html', a)
        # shap.summary_plot(shap_values, training_df.iloc[111], plot_type="bar")
        # plt.savefig('summary_plot.png')
        # show plot
        # shap.force_plot(explainer.expected_value, shap_values[0], training_df)
        # shap.embedding_plot(FEATURE_TARGET, shap_values=shap_values[FEATURE_TARGET])
        # plt.savefig('embedding_plot.png')


        # X_test = training_df.drop(['trace_id', 'label'], 1)
        #
        # X_output = X_test.copy()
        #
        # X_output.loc[:,'predict'] = np.round(model.predict(X_output),2)
        #
        # S = X_output.iloc[1]
        # S
