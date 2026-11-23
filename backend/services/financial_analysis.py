
import math
from typing import Any


def _is_finite_number(value: Any) -> bool:
    """Return True for finite numeric values, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def safe_divide(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    """Return a ratio, or None when inputs are invalid or unavailable."""
    if not _is_finite_number(numerator):
        return None

    if not _is_finite_number(denominator) or denominator == 0:
        return None

    result = numerator / denominator

    return result if math.isfinite(result) else None


def calculate_financial_metrics(
    financials: dict[str, float | None],
) -> dict[str, Any]:
    """Calculate financial metrics from normalized statement values."""

    revenue = financials.get("revenue")
    gross_profit = financials.get("gross_profit")
    operating_income = financials.get("operating_income")
    net_income = financials.get("net_income")

    current_assets = financials.get("current_assets")
    current_liabilities = financials.get("current_liabilities")

    total_debt = financials.get("total_debt")
    total_equity = financials.get("total_equity")
    total_assets = financials.get("total_assets")

    operating_cash_flow = financials.get("operating_cash_flow")
    capital_expenditure = financials.get("capital_expenditure")

    metrics = {
        "operating_margin": safe_divide(operating_income, revenue),
        "net_profit_margin": safe_divide(net_income, revenue),
        "current_ratio": safe_divide(current_assets, current_liabilities),
        "debt_to_equity": safe_divide(total_debt, total_equity),
        "debt_to_assets": safe_divide(total_debt, total_assets),
    }

    # Convert margins to percentages for easier interpretation.
    for metric_name in ("operating_margin", "net_profit_margin"):
        value = metrics[metric_name]
        if value is not None:
            metrics[metric_name] = round(value * 100, 2)

    # Round ratios while preserving unavailable values.
    for metric_name in ("current_ratio", "debt_to_equity", "debt_to_assets"):
        value = metrics[metric_name]
        if value is not None:
            metrics[metric_name] = round(value, 2)

    return {
        "currency": financials.get("currency", "unspecified"),
        "units": financials.get("units", "unspecified"),
        "metrics": metrics,
        "interpretation_notes": [
            "Margins are expressed as percentages.",
            "Ratios are rounded to two decimal places.",
            "Unavailable metrics are returned as null, not guessed.",
            "Interpret ratios in the context of the company's industry "
            "and accounting policies.",
        ],
    }
