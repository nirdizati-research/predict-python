from django.core.management.base import BaseCommand
from pdpbox.utils import _get_grids
from sklearn.externals import joblib
from pdpbox import pdp, get_dataset, info_plots
import matplotlib.pyplot as plt
import numpy as numpy
from src.core.core import get_encoded_logs
from src.encoding.common import retrieve_proper_encoder
from src.encoding.models import ValueEncodings
from src.jobs.models import Job


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        # get model
        TARGET_MODEL = 59
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)[0]
        # load data
        training_df, test_df = get_encoded_logs(job)
        training_df['label'] = training_df['label'].astype(bool).astype(int)
        columns = list(training_df.columns.values)
        features = list(training_df.drop(
            ['trace_id', 'label'],
            1).columns.values)
        feature = 'Age_1'
        feature_grids, percentile_info = _get_grids(
            feature_values=training_df[feature].values, num_grid_points=10, grid_type=None,
            percentile_range='percentile', grid_range=None)
        custom_grids = []
        indexs = []
        for x in range(int(feature_grids.min()), int(feature_grids.max() - 1)):
            custom_grids.append(x);
        print(features)
        fig, axes, summary_df = info_plots.target_plot(
            df=training_df,
            feature=feature,
            feature_name='feature value',
            cust_grid_points=custom_grids,
            target='label', show_percentile=False
        )
        fig.savefig('ice_plot_train_1_3_CType.png')

        lists = list(training_df[feature].values)
        for x in range(int(feature_grids.min()), int(feature_grids.max() - 1)):
            indexs.append(lists.index(x))
        encoder = retrieve_proper_encoder(job)
        encoder.decode(training_df, job.encoding)
        values = training_df[feature].values
        training_df
        lst = []
        print(summary_df)
        if job.encoding.value_encoding != ValueEncodings.BOOLEAN.value:
            for x in range(len(indexs) - 1):
                lst.append({'value': values[indexs[x]],
                            'label': summary_df['label'][x],
                            'count': summary_df['count'][x],
                            })
        else:
            for x in range(summary_df.shape[0]):
                lst.append({'value': summary_df['display_column'][x],
                            'label': summary_df['label'][x],
                            'count': summary_df['count'][x],
                            })
        print(lst)

