import os
import signal
import subprocess

from time import sleep

from django.test.testcases import TestCase

from src.jobs.models import JobTypes, Job
from src.jobs.tasks import prediction_task
from src.runtime.tasks import runtime_task, replay_task
from src.split.models import SplitTypes, SplitOrderingMethods
from src.utils.django_orm import duplicate_orm_row
from src.utils.tests_utils import create_test_job, create_test_split, create_test_log


class TestRuntime(TestCase):

    def test_replay(self):
        pro = subprocess.Popen('python3 manage.py runserver 0.0.0.0:8000', shell=True, stdout=subprocess.PIPE)
        sleep(10)

        job = create_test_job()
        runtime_job = duplicate_orm_row(job)

        runtime_log = create_test_log(log_name='runtime_example.xes',
                                      log_path='cache/log_cache/test_logs/runtime_test.xes')
        runtime_job.split = create_test_split(split_type=SplitTypes.SPLIT_DOUBLE.value,
                                              split_ordering_method=SplitOrderingMethods.SPLIT_SEQUENTIAL.value,
                                              train_log=runtime_log,
                                              test_log=runtime_log)

        replay_task(runtime_job, job)
        self.assertEqual(len(Job.objects.filter(type=JobTypes.REPLAY_PREDICT.value)), 0)
        # TODO: check amount of post request sent to server
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

    def test_runtime(self):
        job = create_test_job()
        runtime_log = create_test_log(log_name='runtime_example.xes',
                                      log_path='cache/log_cache/test_logs/runtime_test.xes')
        runtime_split = create_test_split(split_type=SplitTypes.SPLIT_DOUBLE.value,
                                          split_ordering_method=SplitOrderingMethods.SPLIT_SEQUENTIAL.value,
                                          train_log=runtime_log,
                                          test_log=runtime_log)
        job.create_models = True
        job.save()
        prediction_task(job.id)
        job.refresh_from_db()
        job.split = runtime_split

        runtime_task(job)
