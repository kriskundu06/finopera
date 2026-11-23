
from fastapi.testclient import TestClient

import backend.main as main_module

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



def test_financial_extract_returns_safe_error_when_provider_is_unavailable(monkeypatch):
    def fail_extraction(**kwargs):
        raise RuntimeError("Gemini request failed: private provider detail")

    monkeypatch.setattr(main_module, "extract_financials", fail_extraction)
    response = client.post(
        "/financial-extract",
        params={"document_id": "a" * 64},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The financial extraction provider could not complete the request. Please retry."
    )
    assert "private provider detail" not in response.text


def test_financial_analysis_returns_safe_error_when_provider_is_unavailable(monkeypatch):
    def fail_extraction(**kwargs):
        raise RuntimeError("Gemini request failed: private provider detail")

    monkeypatch.setattr(main_module, "extract_financials", fail_extraction)
    response = client.post(
        "/financial-analyze-document",
        params={"document_id": "b" * 64},
    )

    assert response.status_code == 502
    assert "private provider detail" not in response.text


def test_ask_returns_safe_error_when_provider_is_unavailable(monkeypatch):
    def fail_answer(**kwargs):
        raise RuntimeError("The AI answer provider could not complete the request.")

    monkeypatch.setattr(main_module, "answer_question", fail_answer)
    response = client.post(
        "/ask",
        json={
            "question": "Summarize the report",
            "document_id": "c" * 64,
            "top_k": 5,
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "AI Analyst could not complete the request. Please try again."
    )


def _comparison_report(period, revenue, net_income, currency="USD", units="millions"):
    return {
        "period": period,
        "filename": f"{period}.pdf",
        "currency": currency,
        "units": units,
        "financials": {
            "revenue": revenue,
            "operating_income": revenue * 0.2,
            "net_income": net_income,
            "current_assets": 300,
            "current_liabilities": 150,
            "total_debt": 100,
            "total_equity": 400,
            "total_assets": 600,
        },
    }


def test_multi_period_comparison_calculates_changes():
    response = client.post("/financial-analysis/compare", json={
        "reports": [
            _comparison_report("FY 2024", 1000, 100),
            _comparison_report("FY 2025", 1250, 150),
        ]
    })
    assert response.status_code == 200
    body = response.json()
    assert body["series"]["revenue"] == [
        {"period": "FY 2024", "value": 1000},
        {"period": "FY 2025", "value": 1250},
    ]
    assert body["changes"]["revenue"][0]["absolute_change"] == 250
    assert body["changes"]["revenue"][0]["percentage_change"] == 25
    assert any("increased by 25.00%" in item["text"] for item in body["insights"])


def test_multi_period_comparison_rejects_currency_or_unit_mismatch():
    response = client.post("/financial-analysis/compare", json={
        "reports": [
            _comparison_report("FY 2024", 1000, 100),
            _comparison_report("FY 2025", 1200, 120, currency="INR"),
        ]
    })
    assert response.status_code == 400
    assert "same currency and units" in response.json()["detail"]


def test_multi_period_comparison_rejects_duplicate_periods():
    response = client.post("/financial-analysis/compare", json={
        "reports": [
            _comparison_report("FY 2025", 1000, 100),
            _comparison_report("fy 2025", 1200, 120),
        ]
    })
    assert response.status_code == 400
    assert "duplicated" in response.json()["detail"]


def test_multi_period_comparison_does_not_treat_missing_as_zero():
    first = _comparison_report("FY 2024", 1000, 100)
    second = _comparison_report("FY 2025", 1200, 120)
    first["financials"]["total_debt"] = None
    response = client.post("/financial-analysis/compare", json={"reports": [first, second]})
    assert response.status_code == 200
    assert response.json()["series"]["total_debt"][0]["value"] is None
    assert response.json()["changes"]["total_debt"][0]["percentage_change"] is None


def test_comparison_pdf_export_returns_pdf():
    response = client.post("/financial-analysis/compare/report", json={
        "reports": [
            _comparison_report("FY 2024", 1000, 100),
            _comparison_report("FY 2025", 1200, 120),
        ],
        "currency": "USD",
        "units": "millions",
        "changes": {},
    })
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
