"""Tests for financial-statement page candidate selection."""

import tempfile
import unittest
from pathlib import Path

from backend.services.financial_extraction import _select_statement_pages


class StatementPageSelectionTests(unittest.TestCase):
    def test_relevant_text_pages_rank_before_visual_only_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "scanned-page.png"
            image.write_bytes(b"test")
            pages = [
                {"page": 1, "text": "", "image_path": str(image)},
                {"page": 2, "text": "Revenue from operations 1250", "image_path": None},
            ]
            selected = _select_statement_pages(
                pages,
                {1: 0.0, 2: 9.0},
                "income_statement",
            )
            self.assertEqual(selected[0]["page"], 2)

    def test_visual_pages_are_limited_to_four_fallback_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = []
            for page_number in range(1, 8):
                image = Path(directory) / f"page-{page_number}.png"
                image.write_bytes(b"test")
                pages.append({
                    "page": page_number,
                    "text": f"Scanned page {page_number}",
                    "image_path": str(image),
                })
            selected = _select_statement_pages(
                pages,
                {page["page"]: 0.0 for page in pages},
                "income_statement",
            )
            self.assertEqual(len(selected), 4)

    def test_relevant_pages_are_not_displaced_by_visual_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "visual.png"
            image.write_bytes(b"test")
            pages = [
                {"page": 1, "text": "Revenue 100", "image_path": None},
                {"page": 2, "text": "Scanned page", "image_path": str(image)},
            ]
            selected = _select_statement_pages(
                pages,
                {1: 5.0, 2: 0.0},
                "income_statement",
            )
            self.assertEqual([page["page"] for page in selected], [1, 2])

    def test_balance_sheet_allows_twelve_relevant_pages(self):
        pages = [
            {"page": page_number, "text": "Balance sheet total assets", "image_path": None}
            for page_number in range(1, 15)
        ]
        selected = _select_statement_pages(
            pages,
            {page["page"]: float(page["page"]) for page in pages},
            "balance_sheet",
        )
        self.assertEqual(len(selected), 12)
        self.assertEqual(selected[0]["page"], 14)

    def test_income_statement_limit_is_eight_pages(self):
        pages = [
            {"page": page_number, "text": "Revenue 100", "image_path": None}
            for page_number in range(1, 12)
        ]
        selected = _select_statement_pages(
            pages,
            {page["page"]: float(page["page"]) for page in pages},
            "income_statement",
        )
        self.assertEqual(len(selected), 8)


if __name__ == "__main__":
    unittest.main()
