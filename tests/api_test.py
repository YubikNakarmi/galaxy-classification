from fastapi.testclient import TestClient
from api.classify.routers import router
from fastapi import FastAPI
from api.main import app
import pytest
from core.database import get_db


client = TestClient(app)

@pytest.fixture
class consts:
    valid_payload = {
        "image_b64": ["iVBORw0KGgoAAAANSUhEUgAAAAUA..."],  # truncated base64 string
        "variant": "default"
    }



@pytest.fixture
def db_connection():
    conn = get_db()  # Assuming get_db is a function that returns a database connection
    yield conn          # test runs here
    conn.close() 


def test_celery_task_submission(db_connection,consts):
    # Prepare a valid request payload
    payload = consts.valid_payload

    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data


def test_classify_endpoint(db_connection,consts):
    # Prepare a valid request payload
    payload = consts.valid_payload  

    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data

