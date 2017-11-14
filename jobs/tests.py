from django.test import TestCase

from jobs.models import Job


class LogModelTest(TestCase):
    def setUp(self):
        self.config = {'key': 123}
        Job.objects.create(config=self.config)

    def test_default(self):
        job = Job.objects.get(id=1)

        self.assertEqual(self.config, job.config)
        self.assertEqual('created', job.status)
        self.assertIsNotNone(job.created_date)
        self.assertIsNotNone(job.modified_date)
        self.assertEqual({}, job.result)

    def test_modified(self):
        job = Job.objects.get(id=1)
        job.status = 'completed'

        self.assertNotEquals(job.created_date, job.modified_date)
