"""Regression tests for explicit financial-statement title ranking."""

import unittest

from backend.services.financial_extraction import _statement_title_score


class StatementTitleScoreTests(unittest.TestCase):
    def test_income_statement_title_receives_strong_score(self):
        self.assertEqual(
            _statement_title_score(
                "Annexure D - Statement of consolidated profit and loss",
                "income_statement",
            ),
            30.0,
        )

    def test_balance_sheet_title_receives_strong_score(self):
        self.assertEqual(
            _statement_title_score(
                "Annexure E - Statement of consolidated balance sheet",
                "balance_sheet",
            ),
            30.0,
        )

    def test_cash_flow_title_receives_strong_score(self):
        self.assertEqual(
            _statement_title_score(
                "Annexure F - Statement of consolidated cash flows",
                "cash_flow",
            ),
            30.0,
        )

    def test_unrelated_page_does_not_receive_title_score(self):
        self.assertEqual(
            _statement_title_score(
                "Management discussion and business highlights",
                "balance_sheet",
            ),
            0.0,
        )

    def test_title_matching_is_case_insensitive(self):
        self.assertEqual(
            _statement_title_score(
                "STATEMENT OF CONSOLIDATED BALANCE SHEET",
                "balance_sheet",
            ),
            30.0,
        )


if __name__ == "__main__":
    unittest.main()
