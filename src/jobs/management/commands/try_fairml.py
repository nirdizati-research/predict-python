import matplotlib
from django.core.management.base import BaseCommand

# temporary work around down to virtualenv
# matplotlib issue.
from sklearn.tree import DecisionTreeClassifier

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
from sklearn.ensemble import RandomForestClassifier

class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        plt.style.use('ggplot')
        plt.figure(figsize=(6, 6))
        TARGET_MODEL = 59
        job = Job.objects.filter(pk=TARGET_MODEL)[0]

        training_df, test_df = get_encoded_logs(job)

        X_train = training_df.drop(['trace_id', 'label'], 1)
        RF = DecisionTreeClassifier()

        Y_train = training_df['label'].values
        RF.fit(X_train, Y_train)

        importancies, _ = audit_model(RF.predict, X_train)
        importancies
        print(importancies)

        # generate feature dependence plot
        fig = plot_dependencies(
            importancies.median(),
            reverse_values=False,
            title="FairML feature dependence plot"
        )

        file_name = "fairml_plot_train_1_3_decision_tree.png"
        plt.savefig(file_name, transparent=False, bbox_inches='tight', dpi=550)
