import shap
import sklearn
import xgboost

from django.core.management.base import BaseCommand
from sklearn.externals import joblib

from src.core.core import get_encoded_logs
from src.jobs.models import Job
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import xgboost


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        # get model
        TARGET_MODEL = 5
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)
        model = model[0]

        # load data
        training_df, test_df = get_encoded_logs(job)
        enc = sklearn.preprocessing.OneHotEncoder()
        enc.fit(training_df)
        onehotlabels_train = enc.transform(training_df).toarray()
        enc.fit(test_df);
        onehotlabels_test = enc.transform(test_df).toarray()
        # get radom point in evaluation set
        EXPLANATION_TARGET = 1
        FEATURE_TARGET = 2
        shap.initjs()

        explainer = shap.TreeExplainer(model)
        # explainer
        # shap_values = explainer.shap_values(onehotlabels_train)
        shap_values = explainer.shap_values(training_df)
        # shap_values
        # shap_values1
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        # z = shap.force_plot(explainer.expected_value[0], shap_values[0][1, :], training_df.iloc[1, :],show=False,matplotlib=True).savefig('scratch.png')
        # shap.summary_plot(shap_values, onehotlabels_train,plot_type="bar")
        #
        # shap.force_plot(explainer.expected_value[0], shap_values[0])
        # plt.savefig('force_plot.png')
        shap.initjs()
        a = shap.force_plot(explainer.expected_value[1], shap_values[0], training_df)
        shap.save_html('explainer.html', a)
        a
        html = a.data
        with open('html_file.html', 'w') as f:
            f.write(html)
        plt.savefig('force_plot.png')

        # show plot
        # shap.summary_plot(shap_values, training_df)
        # shap.force_plot(explainer.expected_value, shap_values[0], training_df)
        # shap.embedding_plot(FEATURE_TARGET, shap_values=shap_values[FEATURE_TARGET])

        #
        # #TODO not yet working
        # shap.force_plot(explainer.expected_value, shap_values[EXPLANATION_TARGET, :], training_df.iloc[EXPLANATION_TARGET, :])
        # shap.force_plot(explainer.expected_value, shap_values, training_df)
        # shap.dependence_plot("RM", shap_values, training_df)
        # shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], test_df.iloc[0, :], link="logit") #TODO subst with EXPLANATION_TARGET
        # shap.force_plot(explainer.expected_value[0], shap_values[0], test_df, link="logit")

        # train XGBoost model
        # X, y = shap.datasets.boston()
        # model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
        # #
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(X)
        # print(explainer.expected_value)
        # explainer
        # # shap.force_plot(explainer.expected_value, shap_values, X)
        # # plt.savefig('force_plost.png')
        # # shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :],show=False,matplotlib=True).savefig('scratch.png')
        # shap.dependence_plot("RM", shap_values, X)
        # plt.savefig('scratch.png')
        # print('done')
