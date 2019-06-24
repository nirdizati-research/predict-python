import sys
import unittest
import mock

from manage import manage


class MyTestCase(unittest.TestCase):
    def test_raise_import_error(self):
        with mock.patch.dict(sys.modules, {'django.core.management': None, 'django': None}):
            self.assertRaises(ImportError, manage)

    def test_no_raise_import_error(self):
        with mock.patch.dict(sys.modules, {'django.core.management': None}):
            self.assertRaises(ImportError, manage)

