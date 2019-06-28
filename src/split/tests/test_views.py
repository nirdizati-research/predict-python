from rest_framework.test import APITestCase, APIClient

from src.utils.tests_utils import create_test_split


class TestViews(APITestCase):
    def test_get_split_list(self):
        split = create_test_split()
        client = APIClient()
        response = client.get('/splits/' + str(split.id))
        self.assertEqual(split.id, response.data['id'])
        self.assertEqual(split.type, response.data['type'])
