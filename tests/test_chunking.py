import pytest

from backend.services.chunking import chunk_pages


def test_chunk_pages_preserves_table_rows_and_page_metadata():
    pages = [
        {
            "page": 3,
            "text": (
                "Consolidated Statements\n"
                "Revenue       1,250\n"
                "Cost of revenue 700\n"
                "Gross profit  550"
            ),
        }
    ]

    chunks = chunk_pages(
        pages,
        filename="annual-report.pdf",
        document_id="a" * 64,
        chunk_size=45,
        overlap=5,
    )

    combined = "\n".join(chunk["text"] for chunk in chunks)
    assert "Revenue       1,250" in combined
    assert "Cost of revenue 700" in combined
    assert "Gross profit  550" in combined
    assert all(chunk["metadata"]["page"] == 3 for chunk in chunks)
    assert all(len(chunk["text"]) <= 45 for chunk in chunks)


def test_chunk_pages_splits_an_oversized_table_row():
    long_row = "Operating cash flow " + ("amount " * 20)
    chunks = chunk_pages(
        [{"page": 1, "text": long_row}],
        filename="report.pdf",
        document_id="b" * 64,
        chunk_size=32,
        overlap=4,
    )

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 32 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunk_pages_rejects_invalid_chunk_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(
            [{"page": 1, "text": "Some financial text"}],
            filename="report.pdf",
            document_id="c" * 64,
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_chunk_pages_rejects_invalid_page_text():
    with pytest.raises(ValueError, match="Page text must be a string"):
        chunk_pages(
            [{"page": 1, "text": None}],
            filename="report.pdf",
            document_id="d" * 64,
        )
