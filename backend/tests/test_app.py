import json
import unittest
from unittest.mock import patch

from app import app


class TestFlaskApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.access_token = 'admin'

    def test_version(self):
        response = self.app.get('/api/version')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['version'], '1.0.0')

    @patch('routers.admin.UserModel.find_by_account')
    def test_admin_whoami(self, mock_find_by_account):
        response = self.app.get('/api/admin/whoami')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['msg'], 'admin')


if __name__ == '__main__':
    unittest.main()
