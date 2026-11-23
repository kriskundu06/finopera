"""Regression test for nullable values in financial evidence validation."""

import unittest
from unittest.mock import patch

from backend.services.financial_extraction import _extract_statement


class NullEvidenceValidationTests(unittest.TestCase):
    @patch(
        "backend.services.financial_extraction._call_gemini",
        return_value={
            "revenue": None,
            "evidence": {
                "revenue": {
                    "source_id": "S1",
                    "quote": "Revenue from operations 100",
                }
            },
        },
    )
    def test_null_numeric_value_with_quote_does_not_crash(self, _mock_gemini):
        passages = [{
            "source": "sample-report.pdf",
            "page": 2,
            "text": "Revenue from operations 100",
        }]

        values, evidence, _source_map, statuses = _extract_statement(
            "income_statement",
            passages,
        )

        self.assertIsNone(values["revenue"])
        self.assertIsNone(evidence["revenue"])
        self.assertEqual(statuses["revenue"], "not_found_in_retrieved_text")


if __name__ == "__main__":
    unittest.main()
