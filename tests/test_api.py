from fastapi.testclient import TestClient
from app.api import app   # adjust path if needed

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():
    response = client.post("/predict", params={"video_url": "https://youtube.com/test"})
    assert response.status_code == 200