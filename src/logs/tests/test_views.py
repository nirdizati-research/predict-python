from rest_framework.test import APITestCase, APIClient

from src.utils.tests_utils import create_test_log


class TestViews(APITestCase):
    def test_get_log_list(self):
        create_test_log()
        client = APIClient()
        response = client.get('/logs/')
        self.assertEqual(1, len(response.data))

    def test_get_log_detail(self):
        log = create_test_log()
        client = APIClient()
        response = client.get('/logs/' + str(log.id))
        self.assertEqual(log.id, response.data['id'])
        self.assertEqual(log.name, response.data['name'])
