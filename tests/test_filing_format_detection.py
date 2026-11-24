from backend.services.financial_extraction import _detect_filing_format


def test_detects_indian_ind_as_quarterly_results():
    passages = [{
        "text": "Consolidated Statement of Profit and Loss Q2 FY27 Revenue from operations "
                "Profit for the period Other equity INR in crores Schedule III",
    }]
    assert _detect_filing_format(passages) == "india_ind_as"


def test_detects_us_gaap_sec_style():
    passages = [{
        "text": "Consolidated Statements of Operations Common stock Retained earnings "
                "Form 10-K United States generally accepted accounting principles",
    }]
    assert _detect_filing_format(passages) == "us_gaap_sec"


def test_unknown_format_is_not_guessed():
    passages = [{"text": "Annual report financial highlights and management discussion"}]
    assert _detect_filing_format(passages) == "unknown"


def test_indian_format_requires_multiple_indian_signals():
    passages = [{"text": "Revenue from operations increased during the year"}]
    assert _detect_filing_format(passages) == "unknown"
