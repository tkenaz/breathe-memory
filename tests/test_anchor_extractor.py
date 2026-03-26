"""Tests for AnchorExtractor."""
import pytest
from breathe.anchor_extractor import AnchorExtractor, AnchorResult


def make_extractor(concepts=None):
    return AnchorExtractor(known_concepts=concepts or {})


def test_known_concept_match():
    extractor = make_extractor({"FastAPI": "uuid-001", "Redis": "uuid-002"})
    result = extractor.extract("How do I add caching to FastAPI?")
    entity_texts = [a.text.lower() for a in result.entity_anchors]
    assert "fastapi" in entity_texts


def test_known_concept_sets_node_id():
    extractor = make_extractor({"FastAPI": "uuid-001"})
    result = extractor.extract("I am working on FastAPI")
    matched = [a for a in result.anchors if a.matched_node_id]
    assert len(matched) >= 1
    assert matched[0].matched_node_id == "uuid-001"


def test_temporal_en():
    extractor = make_extractor()
    result = extractor.extract("remember when we fixed the bug yesterday?")
    temporal = [a for a in result.anchors if a.anchor_type == "temporal"]
    assert len(temporal) >= 1


def test_technical_anchor():
    extractor = make_extractor()
    result = extractor.extract("check the routes.py file for the API handler")
    tech = [a for a in result.anchors if a.anchor_type == "technical"]
    assert any("routes.py" in a.text for a in tech)


def test_deduplication():
    extractor = make_extractor({"FastAPI": "uuid-001"})
    # FastAPI appears twice — should deduplicate
    result = extractor.extract("FastAPI is great, I love FastAPI")
    fastapi_anchors = [a for a in result.anchors if "fastapi" in a.text.lower()]
    assert len(fastapi_anchors) == 1


def test_conversation_mode_work():
    extractor = make_extractor()
    result = extractor.extract("class MyModel: def __init__(): import asyncpg from FastAPI")
    assert result.conversation_mode == "work"


def test_conversation_mode_casual():
    extractor = make_extractor()
    result = extractor.extract("hey")
    assert result.conversation_mode == "casual"


def test_empty_message():
    extractor = make_extractor()
    result = extractor.extract("")
    assert isinstance(result, AnchorResult)
    assert len(result.anchors) == 0


def test_node_ids_property():
    extractor = make_extractor({"Redis": "uuid-redis"})
    result = extractor.extract("I need Redis for caching")
    assert "uuid-redis" in result.node_ids
