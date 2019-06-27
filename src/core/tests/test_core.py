import os

from django.test.testcases import TestCase
from pandas.util.testing import assert_frame_equal

from src.cache.cache import get_digested
from src.cache.models import LabelledLog, LoadedLog
from src.core.core import get_encoded_logs
from src.utils.tests_utils import create_test_job


class TestCore(TestCase):
    def test_get_encoded_logs_cache(self):
        job = create_test_job()

        w_cache = get_encoded_logs(job, True)
        wout_cache = get_encoded_logs(job, False)

        assert_frame_equal(w_cache[0], wout_cache[0])
        assert_frame_equal(w_cache[1], wout_cache[1])

        loaded_from_cache = get_encoded_logs(job, True)

        assert_frame_equal(w_cache[0], loaded_from_cache[0])
        assert_frame_equal(w_cache[1], loaded_from_cache[1])

    def test_get_encoded_logs_labeled_cache(self):
        job = create_test_job()

        w_cache = get_encoded_logs(job, True)

        cached_labelled_log = LabelledLog.objects.filter(split=job.split,
                                      encoding=job.encoding,
                                      labelling=job.labelling)[0]

        cached_train = cached_labelled_log.train_log_path
        cached_test = cached_labelled_log.test_log_path

        os.remove('cache/labeled_log_cache/' + get_digested(cached_train) + '.pickle')

        loaded_from_cache = get_encoded_logs(job, True)

        assert_frame_equal(w_cache[0], loaded_from_cache[0])
        assert_frame_equal(w_cache[1], loaded_from_cache[1])

        os.remove('cache/labeled_log_cache/' + get_digested(cached_test) + '.pickle')

        loaded_from_cache = get_encoded_logs(job, True)

        assert_frame_equal(w_cache[0], loaded_from_cache[0])
        assert_frame_equal(w_cache[1], loaded_from_cache[1])

        os.remove('cache/labeled_log_cache/' + get_digested(cached_train) + '.pickle')
        os.remove('cache/labeled_log_cache/' + get_digested(cached_test) + '.pickle')

        loaded_from_cache = get_encoded_logs(job, True)

        assert_frame_equal(w_cache[0], loaded_from_cache[0])
        assert_frame_equal(w_cache[1], loaded_from_cache[1])

    def test_get_encoded_logs_Loaded_cache(self):
        job = create_test_job()

        w_cache = get_encoded_logs(job, True)

        cached_loaded_log = LoadedLog.objects.filter(split=job.split)[0]

        cached_train = cached_loaded_log.train_log_path
        cached_test = cached_loaded_log.test_log_path

        os.remove('cache/loaded_log_cache/' + get_digested(cached_train) + '.pickle')

        loaded_from_cache = get_encoded_logs(job, True)

        assert_frame_equal(w_cache[0], loaded_from_cache[0])
        assert_frame_equal(w_cache[1], loaded_from_cache[1])

        os.remove('cache/loaded_log_cache/' + get_digested(cached_test) + '.pickle')

        loaded_from_cache = get_encoded_logs(job, True)

        assert_frame_equal(w_cache[0], loaded_from_cache[0])
        assert_frame_equal(w_cache[1], loaded_from_cache[1])
