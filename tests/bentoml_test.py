import pytest
import requests

from tests.api_test import consts

@pytest.fixture
def bentoml_client():
    return BentomlClient()


class BentomlClient:
    base_url = "http://localhost:3000"  

    def classify(self, payload):
        response = requests.post(f"{self.base_url}/default", json=payload)
        return response


def test_bentoml_connection():
