from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
CUSTOMER = {
    "age": 35, "income": 85000, "spending_score": 70,
    "membership_years": 6, "purchase_frequency": 30, "last_purchase_amount": 650,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_segments():
    data = client.get("/segments").json()
    assert len(data["segments"]) == 4
    assert sum(s["size"] for s in data["segments"]) == data["n_customers"]


def test_predict():
    r = client.post("/predict", json=CUSTOMER)
    assert r.status_code == 200
    assert r.json()["segment_id"] in range(4)


def test_predict_validation():
    assert client.post("/predict", json={**CUSTOMER, "age": 5}).status_code == 422


def test_predict_batch():
    r = client.post("/predict/batch", json=[CUSTOMER, {**CUSTOMER, "spending_score": 5}])
    assert r.status_code == 200 and len(r.json()) == 2


def test_predict_csv():
    csv = Path(__file__).parent.parent / "data" / "customer_segmentation_data.csv"
    r = client.post("/predict/csv", files={"file": ("c.csv", csv.read_bytes(), "text/csv")})
    assert r.status_code == 200
    assert r.json()["rows"] == 1000


def test_predict_csv_missing_columns():
    r = client.post("/predict/csv", files={"file": ("c.csv", b"age,income\n30,1000\n", "text/csv")})
    assert r.status_code == 400
