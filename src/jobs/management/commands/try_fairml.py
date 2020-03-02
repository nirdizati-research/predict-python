import matplotlib
from django.core.management.base import BaseCommand

# temporary work around down to virtualenv
# matplotlib issue.
from src.core.core import get_encoded_logs

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.linear_model import LogisticRegression

# import specific projection format.
from fairml import audit_model
from fairml import plot_dependencies
from src.jobs.models import Job
from sklearn.externals import joblib

class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        plt.style.use('ggplot')
        plt.figure(figsize=(6, 6))
        TARGET_MODEL = 5
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)[0]


        training_df, test_df = get_encoded_logs(job)

        features = list(training_df.drop(['trace_id', 'label'], 1).columns.values)
        X_train = training_df.drop(['trace_id', 'label'], 1)
        Y_train = training_df.drop(
            ['trace_id', 'prefix_1', 'prefix_2', 'prefix_3', 'label'], 1)

        clf = LogisticRegression(penalty='l2', C=0.01)
        clf.fit(X_train, Y_train)

        importancies, _ = audit_model(clf.predict, X_train)

        print(importancies)

        # generate feature dependence plot
        fig = plot_dependencies(
            importancies.median(),
            reverse_values=False,
            title="FairML feature dependence logistic regression model"
        )

        file_name = "fairml_propublica_linear_direct.png"
        plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=250)
