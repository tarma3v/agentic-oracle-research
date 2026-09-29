"""Tests for evidence retrieval"""

from datetime import datetime, timedelta

from src.retrieval import Source, EvidencePacket, ExaOracleRetriever


def test_source_creation():
    """Test Source dataclass creation"""
    source = Source(
        title="Test Article",
        url="https://example.com",
        published_date="2024-01-01",
        text="This is test content"
    )
    assert source.title == "Test Article"
    assert source.url == "https://example.com"
    assert source.text == "This is test content"


def test_evidence_packet_serialization():
    """Test EvidencePacket JSON serialization"""
    packet = EvidencePacket(
        question_id="TEST001",
        question_text="Will X happen?",
        resolution_criteria="Resolves Yes if...",
        resolution_date="2024-12-31T23:59:59Z",
        retrieval_timestamp="2024-01-01T00:00:00Z",
        query_used='What is the result of this question "Will X happen"',
        sources=[
            Source(title="Source 1", url="https://example.com/1", published_date=None, text="Content 1"),
            Source(title="Source 2", url="https://example.com/2", published_date=None, text="Content 2"),
        ]
    )
    
    json_str = packet.to_json()
    assert "TEST001" in json_str
    assert "Will X happen" in json_str
    
    packet2 = EvidencePacket.from_json(json_str)
    assert packet2.question_id == packet.question_id
    assert packet2.num_sources() == 2


def test_question_to_query():

    query = ExaOracleRetriever.question_to_query("Will the Lakers win the 2024 NBA Finals?")
    assert query == 'What is the result of this question "Will the Lakers win the 2024 NBA Finals"'
    
    query2 = ExaOracleRetriever.question_to_query("Will X happen")
    assert query2 == 'What is the result of this question "Will X happen"'


def test_evidence_packet_format():

    packet = EvidencePacket(
        question_id="TEST001",
        question_text="Will X happen?",
        resolution_criteria="Resolves Yes if...",
        resolution_date="2024-12-31T23:59:59Z",
        retrieval_timestamp="2024-01-01T00:00:00Z",
        query_used="test query",
        sources=[Source(title="Test", url="https://test.com", published_date=None, text="Content")]
    )
    
    formatted = packet.format_for_prompt()
    assert "Question: Will X happen?" in formatted
    assert "Resolution Criteria: Resolves Yes if..." in formatted
    assert "Source 1" in formatted


def test_evidence_packet_uses_highlights_when_present():
    packet = EvidencePacket(
        question_id="TEST002",
        question_text="Will Y happen?",
        resolution_criteria="Resolves Yes if...",
        resolution_date="2024-12-31T23:59:59Z",
        retrieval_timestamp="2024-01-01T00:00:00Z",
        query_used="test query",
        sources=[
            Source(
                title="Test",
                url="https://test.com",
                published_date=None,
                text="This text should not be used when highlights exist.",
                highlights=["Highlight A", "Highlight B"],
            )
        ],
    )
    formatted = packet.format_for_prompt()
    assert "- Highlight A" in formatted
    assert "- Highlight B" in formatted


def test_normalize_iso8601_fractional_seconds_padding():
    raw = "2025-02-24T01:15:11.9396+00:00"
    normalized = ExaOracleRetriever._normalize_iso8601(raw)
    assert normalized == "2025-02-24T01:15:11.939600+00:00"
    dt = datetime.fromisoformat(normalized)
    end_date = (dt + timedelta(days=1)).isoformat().replace("+00:00", "Z")
    assert end_date.startswith("2025-02-25T01:15:11.939600")
