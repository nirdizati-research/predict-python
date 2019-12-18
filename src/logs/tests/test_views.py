from rest_framework.test import APITestCase, APIClient

from src.utils.tests_utils import create_test_log
from rest_framework import status, mixins, generics


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

    def test_get_log_traces_attributes(self):
        log = create_test_log()
        client = APIClient()
        response = client.get('/logs/' + str(log.id)+'/traces')
        self.assertEqual(6, len(response.data))
        self.assertEqual({'concept:name': '3', 'creator': 'Fluxicon Nitro'}, response.data[0]['attributes'])
        self.assertEqual('Pete', response.data[0]['events'][0]['Resource'])
        self.assertEqual(9, len(response.data[0]['events']))
        response = client.get('/logs/' + str(-12222112)+'/traces')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
