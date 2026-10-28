
import math

import pytest

from backend.services.financial_analysis import (
    calculate_financial_metrics,
    safe_divide,
)


def test_safe_divide():
    assert safe_divide(10, 2) == 5


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (None, 2),
        (10, None),
        (10, 0),
        (None, None),
    ],
)
def test_safe_divide_returns_none_for_unavailable_inputs(
    numerator,
    denominator,
):
    assert safe_divide(numerator, denominator) is None


def test_financial_metrics_calculated_correctly():
    financials = {
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

    result = calculate_financial_metrics(financials)
    metrics = result["metrics"]

    assert metrics["gross_margin"] == 60.0
    assert metrics["operating_margin"] == 20.0
    assert metrics["net_profit_margin"] == 15.0
    assert metrics["current_ratio"] == 2.0
    assert metrics["debt_to_equity"] == 0.5
    assert metrics["debt_to_assets"] == 0.25
    assert metrics["operating_cash_flow_to_net_income"] == pytest.approx(1.67)
    assert metrics["free_cash_flow"] == 200.0

    assert result["currency"] == "USD"
    assert result["units"] == "millions"


def test_missing_inputs_return_none():
    result = calculate_financial_metrics({})
    metrics = result["metrics"]

    assert metrics["gross_margin"] is None
    assert metrics["current_ratio"] is None
    assert metrics["debt_to_equity"] is None
    assert metrics["debt_to_assets"] is None
    assert metrics["operating_cash_flow_to_net_income"] is None
    assert metrics["free_cash_flow"] is None


def test_zero_denominators_do_not_raise():
    financials = {
        "revenue": 0,
        "gross_profit": 100,
        "operating_income": 20,
        "net_income": 10,
        "current_assets": 100,
        "current_liabilities": 0,
        "total_debt": 50,
        "total_equity": 0,
        "total_assets": 0,
        "operating_cash_flow": 10,
        "capital_expenditure": 2,
    }

    metrics = calculate_financial_metrics(financials)["metrics"]

    assert metrics["gross_margin"] is None
    assert metrics["operating_margin"] is None
    assert metrics["net_profit_margin"] is None
    assert metrics["current_ratio"] is None
    assert metrics["debt_to_equity"] is None
    assert metrics["debt_to_assets"] is None


def test_free_cash_flow_can_be_negative():
    financials = {
        "operating_cash_flow": 100,
        "capital_expenditure": 150,
    }

    metrics = calculate_financial_metrics(financials)["metrics"]

    assert metrics["free_cash_flow"] == -50.0


def test_safe_divide_rejects_nan_and_infinity():
    assert safe_divide(float("nan"), 2) is None
    assert safe_divide(10, float("inf")) is None
    assert safe_divide(float("-inf"), 2) is None


def test_non_finite_cash_flow_returns_none():
    financials = {
        "operating_cash_flow": float("inf"),
        "capital_expenditure": 50,
    }

    metrics = calculate_financial_metrics(financials)["metrics"]

    assert metrics["free_cash_flow"] is None


def test_non_finite_inputs_do_not_contaminate_metrics():
    financials = {
        "revenue": float("nan"),
        "gross_profit": 100,
        "total_debt": 500,
        "total_equity": float("inf"),
    }

    metrics = calculate_financial_metrics(financials)["metrics"]

    assert metrics["gross_margin"] is None
    assert metrics["debt_to_equity"] is None
