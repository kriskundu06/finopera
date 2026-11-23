"""Regression tests for deterministic financial extraction helpers."""

import unittest

from backend.services.financial_extraction import (
    STATEMENT_FIELDS,
    _merge_overlapping_chunks,
    _normalize_text,
    _quote_contains_value,
    _response_schema,
)


class ChunkReconstructionTests(unittest.TestCase):
    def test_merges_shared_boundary_without_duplicate_text(self):
        chunks = [
            {"text": "Revenue from operations 24,138 Total"},
            {"text": "Total income 24,489 Profit after tax 125"},
        ]
        self.assertEqual(
            _merge_overlapping_chunks(chunks),
            "Revenue from operations 24,138 Total income 24,489 Profit after tax 125",
        )

    def test_merges_exact_overlap(self):
        chunks = [
            {"text": "Revenue from operations 24,138"},
            {"text": "24,138 Total income 24,489"},
        ]
        self.assertEqual(
            _merge_overlapping_chunks(chunks),
            "Revenue from operations 24,138 Total income 24,489",
        )

    def test_ignores_empty_chunks_and_handles_empty_input(self):
        self.assertEqual(
            _merge_overlapping_chunks([{"text": ""}, {"text": "  "}, {"text": "Total assets 45,109"}]),
            "Total assets 45,109",
        )
        self.assertEqual(_merge_overlapping_chunks([]), "")


class EvidenceNumberMatchingTests(unittest.TestCase):
    def test_matches_supported_numeric_formats(self):
        cases = [
            (24138, "Revenue from operations 24,138"),
            (125.5, "Profit for the period 125.50"),
            (-627, "Net cash from operations (627)"),
            (-627, "Net cash from operations -627"),
            (0, "Capital expenditure 0"),
        ]
        for value, quote in cases:
            with self.subTest(value=value, quote=quote):
                self.assertTrue(_quote_contains_value(value, quote))

    def test_rejects_wrong_or_partial_amount(self):
        self.assertFalse(_quote_contains_value(24138, "Revenue from operations 24,183"))
        self.assertFalse(_quote_contains_value(250, "Revenue from operations 1,250"))


class ExtractionSchemaTests(unittest.TestCase):
    def test_schema_contains_every_statement_field(self):
        for statement, fields in STATEMENT_FIELDS.items():
            if statement == "reporting_details":
                continue
            schema = _response_schema(statement)
            for field in fields:
                self.assertIn(field, schema)
                self.assertIsNone(schema[field])
                self.assertIn(field, schema["evidence"])

    def test_normalization_ignores_case_and_punctuation(self):
        self.assertEqual(
            _normalize_text("Profit After Tax (PAT)"),
            _normalize_text("profit-after-tax PAT"),
        )


if __name__ == "__main__":
    unittest.main()
