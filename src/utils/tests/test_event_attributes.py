from django.test import TestCase

from src.utils.event_attributes import unique_events, unique_events2, get_event_attributes, \
    get_additional_columns
from src.utils.file_service import get_log
from src.utils.tests_utils import general_example_test_filepath, \
    general_example_train_filepath, general_example_test_filename, create_test_log, general_example_train_filename


class EventAttributes(TestCase):
    def setUp(self):
        self.log = get_log(create_test_log(log_name=general_example_test_filename,
                                           log_path=general_example_test_filepath))

    def test_unique_events(self):
        events = unique_events(self.log)
        self.assertEqual(7, len(events))

    def test_multiple_unique_events(self):
        test_log = get_log(create_test_log(log_name=general_example_test_filename,
                                           log_path=general_example_test_filepath))
        training_log = get_log(create_test_log(log_path=general_example_train_filepath,
                                               log_name=general_example_train_filename))
        events = unique_events2(training_log, test_log)
        self.assertEqual(8, len(events))

    def test_event_attributes(self):
        log = get_log(create_test_log(log_name=general_example_test_filename,
                                      log_path=general_example_test_filepath))
        attributes = get_event_attributes(log)
        self.assertListEqual(attributes, ['Activity', 'Costs', 'Resource', 'org:resource'])

    def test_global_event_attributes(self):
        log = get_log(create_test_log(log_name=general_example_test_filename,
                                      log_path=general_example_test_filepath))
        attributes = get_additional_columns(log)
        self.assertListEqual(attributes['event_attributes'], ['Activity', 'Costs', 'Resource', 'org:resource'])
