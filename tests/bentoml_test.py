import pytest
import requests

from constants import base_64_test
from PIL import Image

@pytest.fixture
def bentoml_client():
    return BentomlClient()


class BentomlClient:
    base_url = "http://localhost:3000"  

    def classify(self, payload):
        response = requests.post(f"{self.base_url}/default", json=payload)
        return response


def test_bentoml_connection():
    client = BentomlClient()
    response = client.classify({"image": base_64_test()})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
