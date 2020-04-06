from rest_framework.test import APITestCase, APIClient

from src.utils.tests_utils import create_test_split, create_test_log


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
