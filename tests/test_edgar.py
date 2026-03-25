"""Tests for SEC EDGAR pipeline."""
from unittest.mock import patch, MagicMock
import pytest

from src.research.edgar_pipeline import search_fulltext, search_filings, SP500_SAMPLE


def test_sp500_sample_has_companies():
    assert len(SP500_SAMPLE) >= 20
    assert "AAPL" in SP500_SAMPLE
    assert "MSFT" in SP500_SAMPLE


def test_search_fulltext_with_mock():
    """Mock EDGAR API response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "display_names": ["APPLE INC (AAPL)"],
                        "form_type": "10-Q",
                        "file_date": "2024-06-15",
                        "entity_id": "320193",
                    },
                    "highlight": {"file_description": ["Revenue guidance exceeded expectations"]},
                },
            ]
        }
    }

    with patch("src.research.edgar_pipeline.requests.get", return_value=mock_resp):
        results = search_fulltext("revenue guidance", max_results=5)
        assert len(results) == 1
        assert "APPLE" in results[0]["company"]


def test_search_fulltext_handles_failure():
    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch("src.research.edgar_pipeline.requests.get", return_value=mock_resp):
        results = search_fulltext("test")
        assert results == []


def test_search_fulltext_handles_exception():
    with patch("src.research.edgar_pipeline.requests.get", side_effect=Exception("timeout")):
        results = search_fulltext("test")
        assert results == []


def test_search_filings_with_mock():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "display_names": ["TESLA INC"],
                        "form_type": "8-K",
                        "file_date": "2024-10-20",
                        "entity_id": "1318605",
                        "file_num": "001-34756",
                    },
                },
            ]
        }
    }

    with patch("src.research.edgar_pipeline.requests.get", return_value=mock_resp):
        results = search_filings("TSLA", form_type="8-K")
        assert len(results) == 1
        assert "TESLA" in results[0]["company"]
        assert results[0]["form_type"] == "8-K"
