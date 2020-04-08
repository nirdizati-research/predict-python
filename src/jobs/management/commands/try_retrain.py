import random

from django.core.management.base import BaseCommand
from pandas import Series

from src.cache.cache import put_labelled_logs
from src.core.core import get_encoded_logs

from src.jobs.models import Job
from src.jobs.tasks import prediction_task
from src.runtime.tasks import create_prediction_job
from src.utils.django_orm import duplicate_orm_row


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        TARGET_JOB = 439
        initial_job_obj = Job.objects.filter(pk=TARGET_JOB)[0]

        # todo: return performances
        print('Initial Job:', initial_job_obj.evaluation.classificationmetrics)  # TODO future bug

        training_df_old, test_df_old = get_encoded_logs(initial_job_obj)
        training_df = training_df_old.copy()
        test_df = test_df_old.copy()

        # todo: what should I randomise?
        for df in [training_df, test_df]:
            m_col = df[["prefix_2", "prefix_4"]]
            del df["prefix_2"]
            del df["prefix_4"]
            target_values1 = list(set(m_col['prefix_2']))
            target_values2 = list(set(m_col['prefix_4']))
            df[["prefix_2", "prefix_4"]] = m_col.apply(
                lambda x:
                    x
                        if ((x[0] != 2) and (x[1] != 3)) else
                    Series({
                        "prefix_2": random.choice(target_values1),
                        "prefix_4": random.choice(target_values2)
                    }),
                axis=1
            )

        assert not training_df.equals(training_df_old)
        assert not test_df.equals(test_df_old)

        # todo: save new dataset in memory and create split to use it
        initial_split_obj = initial_job_obj.split
        new_split = duplicate_orm_row(initial_split_obj)
        train_log = duplicate_orm_row(new_split.train_log)
        test_log = duplicate_orm_row(new_split.test_log)

        # TODO future bug creates shadows
        train_log.name = 'RETRAIN' + train_log.name
        train_log.path = 'cache/log_cache/' + train_log.name
        train_log.properties = {}
        test_log.name = 'RETRAIN' + test_log.name
        test_log.path = 'cache/log_cache/' + test_log.name
        test_log.properties = {}

        new_split.train_log = train_log
        new_split.test_log = test_log
        new_split.additional_columns = None
        new_split.save()

        prediction_job = create_prediction_job(initial_job_obj, initial_job_obj.encoding.prefix_length)
        prediction_job.split = new_split
        prediction_job.split.save()
        prediction_job.save()

        put_labelled_logs(prediction_job, training_df, test_df)

        # todo: build model
        prediction_task(prediction_job.id, do_publish_result=False)
        prediction_job.refresh_from_db()

        # todo: return performances
        print('Retrain Job:', prediction_job.evaluation.classificationmetrics)


        print('Done, cheers!')

