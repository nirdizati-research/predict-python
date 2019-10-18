from django.test.testcases import TestCase

from src.jobs.job_creator import generate
from src.jobs.tasks import prediction_task
from src.logs.models import Log
from src.runtime.tasks import runtime_task, replay_task
from src.split.models import SplitTypes, SplitOrderingMethods
from src.utils.django_orm import duplicate_orm_row
from src.utils.tests_utils import create_test_job, create_test_split


class TestRuntime(TestCase):

    def test_replay(self):
        job = create_test_job()
        runtime_log = create_runtime_log()
        runtime_split = create_test_split(split_type = SplitTypes.SPLIT_DOUBLE.value,
                                          split_ordering_method = SplitOrderingMethods.SPLIT_SEQUENTIAL.value,
                                          train_log= runtime_log,
                                          test_log = runtime_log)

        runtime_job = duplicate_orm_row(job)
        runtime_job.split = runtime_split

        replay_task(runtime_job, job)


def create_runtime_log(log_name: str = 'runtime_example.xes', log_path: str = 'cache/log_cache/test_logs/runtime_test.xes') -> Log:
    log = Log.objects.get_or_create(name=log_name, path=log_path)[0]
    return log
