import numpy as np
import pandas as pd
from rulefit import RuleFit

import sklearn

from django.core.management.base import BaseCommand
from sklearn.externals import joblib

from src.core.core import get_encoded_logs
from src.jobs.models import Job
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        # get model
        TARGET_MODEL = 5
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)
        model = model[0]
        training_df, test_df = get_encoded_logs(job)
        feature_names = list(training_df.drop(['trace_id', 'label'], 1).columns.values)

        X_train = training_df.drop(['trace_id','label'], 1)
        Y_train = training_df.drop(['trace_id', 'prefix_1','prefix_3', 'prefix_4','label'], 1)

        rf = RuleFit()
        columns = list(X_train.columns)

        X = X_train.as_matrix()

        rf.fit(X, Y_train.values.ravel(), feature_names=columns)
        rules = rf.get_rules()
        # rules = rules[rules.coef != 0].sort_values("support", ascending=False)
        rules = rules[(rules.coef > 0.) & (rules.type != 'linear')]
        rules['effect'] = rules['coef'] * rules['support']
        pd.set_option('display.max_colwidth', -1)
        rules.nlargest(10, 'effect')
        # print(rules)
        rules
