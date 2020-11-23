from django.test.testcases import TestCase
from pandas.util.testing import assert_frame_equal

from src.cache.cache import get_labelled_logs
from src.encoding.common import get_encoded_logs
from src.utils.tests_utils import create_test_job


class TestViews(TestCase):
    def test_get_labelled_logs(self):
        job = create_test_job()
        labelled_logs = get_encoded_logs(job)

        cached_labelled_logs = get_labelled_logs(job)

        assert_frame_equal(labelled_logs[0], cached_labelled_logs[0])
        assert_frame_equal(labelled_logs[1], cached_labelled_logs[1])
