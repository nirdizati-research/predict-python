from rest_framework.test import APITestCase, APIClient

from src.logs.tests.test_split import split_double, split_single
from src.split.models import Split, SplitTypes
from src.utils.tests_utils import create_test_split, create_test_log, general_example_train_filepath, \
    general_example_test_filepath_xes


class TestViews(APITestCase):
    def test_get_split_list(self):
        split = create_test_split()
        client = APIClient()
        response = client.get('/splits/' + str(split.id))
        self.assertEqual(split.id, response.data['id'])
        self.assertEqual(split.type, response.data['type'])

    def test_post_split_create_split(self):
        log = create_test_log()
        client = APIClient()
        response = client.post('/splits/', {
            'original_log': log.id,
            'splitting_method': 'sequential',
            'test_size': .2})
        self.assertEqual(log.id, response.data['original_log'])
        self.assertEqual('sequential', response.data['splitting_method'])
        self.assertEqual(.2, response.data['test_size'])

    def test_get_split_train_logs_with_double(self):
        split = split_double()
        client = APIClient()
        response = client.get('/splits/' + str(split.id)+"/logs/train")
        f = open(general_example_train_filepath, "r")
        self.assertEqual(f.read(), response.content.decode())

    def test_get_split_train_logs_with_single(self):
        split = split_single()
        client = APIClient()
        response = client.get('/splits/' + str(split.id)+"/logs/train")
        split_obj = Split.objects.filter(
            type=SplitTypes.SPLIT_DOUBLE.value,
            original_log=split.original_log,
            test_size=split.test_size,
            splitting_method=split.splitting_method
        )[0]
        f = open(split_obj.train_log.path, "r")
        self.assertEqual(f.read(), response.content.decode())

    def test_get_split_test_logs_with_double(self):
        split = split_double()
        client = APIClient()
        response = client.get('/splits/' + str(split.id) + "/logs/test")
        f = open(general_example_test_filepath_xes, "r")
        self.assertEqual(f.read(), response.content.decode())

    def test_get_split_test_logs_with_single(self):
        split = split_single()
        client = APIClient()
        response = client.get('/splits/' + str(split.id) + "/logs/test")
        split_obj = Split.objects.filter(
            type=SplitTypes.SPLIT_DOUBLE.value,
            original_log=split.original_log,
            test_size=split.test_size,
            splitting_method=split.splitting_method
        )[0]
        f = open(split_obj.test_log.path, "r")
        self.assertEqual(f.read(), response.content.decode())

