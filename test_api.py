
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()
    assert body["message"] == "AI Due Diligence Copilot API is running"
    assert "version" in body


def test_financial_analysis_endpoint():
    payload = {
        "revenue": 1000,
        "gross_profit": 600,
        "operating_income": 200,
        "net_income": 150,
        "current_assets": 300,
        "current_liabilities": 150,
        "total_debt": 400,
        "total_equity": 800,
        "total_assets": 1600,
        "operating_cash_flow": 250,
        "capital_expenditure": 50,
        "currency": "USD",
        "units": "millions",
    }

    response = client.post("/financial-analysis", json=payload)

    assert response.status_code == 200

    body = response.json()
    metrics = body["metrics"]

    assert metrics["gross_margin"] == 60.0
    assert metrics["current_ratio"] == 2.0
    assert metrics["debt_to_equity"] == 0.5
    assert metrics["free_cash_flow"] == 200.0


def test_financial_analysis_rejects_invalid_payload():
    response = client.post(
        "/financial-analysis",
        json={"revenue": "not-a-number"},
    )

    assert response.status_code == 422


def test_financial_analysis_handles_missing_values():
    response = client.post("/financial-analysis", json={})

    assert response.status_code == 200

    metrics = response.json()["metrics"]

    assert metrics["gross_margin"] is None
    assert metrics["debt_to_equity"] is None
    assert metrics["free_cash_flow"] is None


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
