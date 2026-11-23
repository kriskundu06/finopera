import pytest

from backend.services.financial_analysis import calculate_financial_metrics, safe_divide


def test_safe_divide():
    assert safe_divide(10, 2) == 5


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [(None, 2), (10, None), (10, 0), (None, None)],
)
def test_safe_divide_returns_none_for_unavailable_inputs(numerator, denominator):
    assert safe_divide(numerator, denominator) is None


def test_core_financial_metrics_calculated_correctly():
    financials = {
        "revenue": 1000,
        "operating_income": 200,
        "net_income": 150,
        "current_assets": 300,
        "current_liabilities": 150,
        "total_debt": 400,
        "total_equity": 800,
        "total_assets": 1600,
        "currency": "USD",
        "units": "millions",
    }
    result = calculate_financial_metrics(financials)
    metrics = result["metrics"]

    assert metrics["operating_margin"] == 20.0
    assert metrics["net_profit_margin"] == 15.0
    assert metrics["current_ratio"] == 2.0
    assert metrics["debt_to_equity"] == 0.5
    assert metrics["debt_to_assets"] == 0.25
    assert set(metrics) == {
        "operating_margin",
        "net_profit_margin",
        "current_ratio",
        "debt_to_equity",
        "debt_to_assets",
    }
    assert result["currency"] == "USD"
    assert result["units"] == "millions"


def test_missing_inputs_return_none():
    metrics = calculate_financial_metrics({})["metrics"]
    assert all(value is None for value in metrics.values())


def test_zero_denominators_do_not_raise():
    financials = {
        "revenue": 0,
        "operating_income": 20,
        "net_income": 10,
        "current_assets": 100,
        "current_liabilities": 0,
        "total_debt": 50,
        "total_equity": 0,
        "total_assets": 0,
    }
    metrics = calculate_financial_metrics(financials)["metrics"]
    assert metrics["operating_margin"] is None
    assert metrics["net_profit_margin"] is None
    assert metrics["current_ratio"] is None
    assert metrics["debt_to_equity"] is None
    assert metrics["debt_to_assets"] is None


def test_safe_divide_rejects_nan_and_infinity():
    assert safe_divide(float("nan"), 2) is None
    assert safe_divide(10, float("inf")) is None
    assert safe_divide(float("-inf"), 2) is None



def test_indian_unit_labels_are_canonicalized_without_conversion():
    from backend.services.financial_extraction import _normalize_units

    assert _normalize_units("₹ in crore") == "crores"
    assert _normalize_units("Rs. Lakhs") == "lakhs"
    assert _normalize_units("INR millions") == "millions"
    assert _normalize_units("in thousands") == "thousands"
    assert _normalize_units(None) == "unspecified"


def test_financial_ratios_are_scale_invariant_for_indian_units():
    crore_result = calculate_financial_metrics({
        "revenue": 1000,
        "operating_income": 200,
        "net_income": 100,
        "current_assets": 300,
        "current_liabilities": 150,
        "total_debt": 400,
        "total_equity": 800,
        "total_assets": 1600,
        "currency": "INR",
        "units": "crores",
    })
    lakh_result = calculate_financial_metrics({
        "revenue": 100000,
        "operating_income": 20000,
        "net_income": 10000,
        "current_assets": 30000,
        "current_liabilities": 15000,
        "total_debt": 40000,
        "total_equity": 80000,
        "total_assets": 160000,
        "currency": "INR",
        "units": "lakhs",
    })

    assert crore_result["metrics"] == lakh_result["metrics"]
    assert crore_result["units"] == "crores"
    assert lakh_result["units"] == "lakhs"


def test_extraction_status_is_partial_when_core_fields_are_missing():
    from backend.services.financial_extraction import NUMERIC_FIELDS, _build_extraction_status

    financials = {"revenue": 24138, "net_income": 125, "total_equity": 31268}
    evidence = {field: {"type": "source_quote"} for field in financials}
    status = _build_extraction_status(financials, evidence, {})

    assert status["extraction_status"] == "partial"
    assert status["verified_fields"] == ["net_income", "revenue", "total_equity"]
    assert set(status["missing_fields"]) == NUMERIC_FIELDS - set(financials)


def test_extraction_status_is_completed_only_when_all_core_fields_are_verified():
    from backend.services.financial_extraction import NUMERIC_FIELDS, _build_extraction_status

    financials = {field: 1 for field in NUMERIC_FIELDS}
    evidence = {field: {"type": "source_quote"} for field in NUMERIC_FIELDS}
    status = _build_extraction_status(financials, evidence, {})

    assert status["extraction_status"] == "completed"
    assert status["missing_fields"] == []


def test_statement_relevance_prefers_financial_statement_tables():
    from backend.services.financial_extraction import _statement_relevance_score

    income = "Consolidated Statement of Profit and Loss Revenue from operations Profit for the period"
    balance = "Consolidated Balance Sheet Total current assets Total current liabilities Total assets Total equity"

    assert _statement_relevance_score(income, "income_statement") >= 3
    assert _statement_relevance_score(balance, "balance_sheet") >= 4
    assert _statement_relevance_score("Chairman's letter and business highlights", "balance_sheet") == 0


def test_document_statement_retrieval_is_scoped_and_prefers_labelled_chunks(monkeypatch):
    import backend.services.financial_extraction as extraction

    document_id = "a" * 64
    chunks = [
        {"document_id": document_id, "page": 22, "chunk_number": 1,
         "text": "Consolidated Balance Sheet Total current assets Total current liabilities Total assets Total equity"},
        {"document_id": document_id, "page": 4, "chunk_number": 1,
         "text": "Chairman's letter and business highlights"},
        {"document_id": "b" * 64, "page": 22, "chunk_number": 1,
         "text": "Consolidated Balance Sheet Total current assets Total current liabilities Total assets Total equity"},
    ]
    monkeypatch.setattr(extraction, "get_document_chunks", lambda _: chunks)

    grouped = extraction._retrieve_statement_passages(document_id)

    assert len(grouped["balance_sheet"]) == 1
    assert grouped["balance_sheet"][0]["page"] == 22
    assert all(
        chunk["document_id"] == document_id
        for passages in grouped.values() for chunk in passages
    )
