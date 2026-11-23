
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


def test_due_diligence_report_endpoint_generates_cited_pdf(monkeypatch):
    def fake_report(**kwargs):
        assert kwargs["document_id"] == "d" * 64
        return {
            "report": (
                "1. Executive summary\\n"
                "DOCUMENT FACT: Revenue was reported in the filing [S1].\\n"
                "11. Overall preliminary risk assessment\\n"
                "Unable to assess from the supplied report."
            ),
            "sources": [
                {"source_id": "S1", "source": "annual-report.pdf", "page": 4, "text": "Revenue"},
            ],
            "model": "test-model",
            "reporting_period": "FY 2025",
        }

    monkeypatch.setattr(main_module, "generate_due_diligence_report", fake_report)
    response = client.post("/due-diligence/report", json={
        "document_id": "d" * 64,
        "filename": "annual-report.pdf",
        "reporting_period": "FY 2025",
        "financials": {"revenue": 100},
        "analysis": {"metrics": {"net_profit_margin": 10}},
    })
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_due_diligence_report_rejects_invalid_document_id():
    response = client.post("/due-diligence/report", json={"document_id": "invalid"})
    assert response.status_code == 422


def test_ask_passes_bounded_conversation_history_to_rag(monkeypatch):
    captured = {}

    def fake_answer(**kwargs):
        captured.update(kwargs)
        return {"answer": "The cited report states this in [S1].", "sources": []}

    monkeypatch.setattr(main_module, "answer_question", fake_answer)
    response = client.post("/ask", json={
        "question": "What does that imply?",
        "document_id": "e" * 64,
        "top_k": 5,
        "conversation_history": [
            {"role": "user", "content": "What is the debt?"},
            {"role": "assistant", "content": "The report lists debt at 100."},
        ],
    })
    assert response.status_code == 200
    assert captured["conversation_history"] == [
        {"role": "user", "content": "What is the debt?"},
        {"role": "assistant", "content": "The report lists debt at 100."},
    ]


def test_ask_rejects_more_than_eight_history_turns():
    response = client.post("/ask", json={
        "question": "Follow up",
        "document_id": "f" * 64,
        "conversation_history": [
            {"role": "user", "content": f"Question {index}"}
            for index in range(9)
        ],
    })
    assert response.status_code == 422
