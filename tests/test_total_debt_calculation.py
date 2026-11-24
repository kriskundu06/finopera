from backend.services.financial_extraction import _calculate_total_debt


def test_calculates_total_debt_from_complete_verified_components():
    components = {
        "short_term_debt": {"value": 1000, "source_id": "S1", "quote": "Short-term debt 1,000"},
        "long_term_debt": {"value": 32366, "source_id": "S2", "quote": "Long-term debt 32,366"},
    }

    total, evidence = _calculate_total_debt(components, components_complete=True)

    assert total == 33366
    assert evidence["type"] == "calculated"
    assert evidence["component_count"] == 2
    assert evidence["inputs"] == components


def test_does_not_calculate_from_incomplete_component_coverage():
    components = {
        "long_term_debt": {"value": 32366, "source_id": "S2", "quote": "Long-term debt 32,366"},
    }

    total, evidence = _calculate_total_debt(components, components_complete=False)

    assert total is None
    assert evidence is None


def test_rejects_negative_or_invalid_debt_components():
    components = {
        "short_term_debt": {"value": -100, "source_id": "S1", "quote": "Debt (100)"},
        "long_term_debt": {"value": 500, "source_id": "S2", "quote": "Debt 500"},
    }

    total, evidence = _calculate_total_debt(components, components_complete=True)

    assert total is None
    assert evidence is None


def test_rejects_unknown_component_types():
    components = {
        "operating_lease_liabilities": {"value": 900, "source_id": "S1", "quote": "Lease liabilities 900"},
    }

    total, evidence = _calculate_total_debt(components, components_complete=True)

    assert total is None
    assert evidence is None
