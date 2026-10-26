from fastapi.testclient import TestClient
from app.main import app

def client():
    return TestClient(app)  # `with` triggers lifespan -> loads model

def test_health():
    with client() as c:
        assert c.get("/health").json()["status"] == "ok"

def test_predict_ok():
    with client() as c:
        r = c.post("/predict", json={"HomePlanet": "Europa", "CryoSleep": True, "Cabin": "B/0/P", "Age": 30})
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["transported"], bool) and 0 <= body["probability"] <= 1

def test_empty_body_is_allowed():
    with client() as c:
        assert c.post("/predict", json={}).status_code == 200

def test_bad_planet_rejected():
    with client() as c:
        r = c.post("/predict", json={"HomePlanet": "Pluto"})
        assert r.status_code == 422 and r.json()["error"] == "Invalid input"

def test_bad_cabin_rejected():
    with client() as c:
        assert c.post("/predict", json={"Cabin": "nonsense"}).status_code == 422

def test_negative_spend_rejected():
    with client() as c:
        assert c.post("/predict", json={"Spa": -5}).status_code == 422

def test_typo_field_rejected():
    with client() as c:
        assert c.post("/predict", json={"CryoSleeep": True}).status_code == 422

def test_batch():
    with client() as c:
        r = c.post("/predict/batch", json={"passengers": [{"Age": 5}, {"CryoSleep": True}]})
        assert r.status_code == 200 and len(r.json()) == 2
