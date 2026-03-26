"""Tests for ContextInjector."""
import pytest
from breathe.context_injector import ContextInjector
from breathe.interfaces import RetrievedNode


def make_node(concept, node_type="entity", summary=None, memory_content=None, importance=0.7):
    return RetrievedNode(
        node_id=f"id-{concept}",
        concept=concept,
        node_type=node_type,
        summary=summary,
        importance=importance,
        depth=0,
        memory_content=memory_content,
    )


def test_returns_none_for_empty_nodes():
    injector = ContextInjector()
    assert injector.format_injection([]) is None


def test_wraps_in_memory_tag():
    injector = ContextInjector()
    nodes = [make_node("Redis", summary="In-memory data store")]
    result = injector.format_injection(nodes)
    assert "<associative_memory>" in result
    assert "</associative_memory>" in result


def test_custom_memory_tag():
    injector = ContextInjector(memory_tag="my_memory")
    nodes = [make_node("Redis", summary="cache")]
    result = injector.format_injection(nodes)
    assert "<my_memory>" in result


def test_entity_concept_in_output():
    injector = ContextInjector()
    nodes = [make_node("PostgreSQL", summary="Relational database with pgvector")]
    result = injector.format_injection(nodes)
    assert "PostgreSQL" in result


def test_memory_content_included():
    injector = ContextInjector()
    nodes = [make_node("Redis", node_type="memory_keyword",
                        memory_content="Redis was discussed last week for session storage")]
    result = injector.format_injection(nodes)
    assert "session storage" in result


def test_respects_budget():
    injector = ContextInjector(mode_budgets={"balanced": 10})  # tiny budget
    nodes = [make_node("Redis", summary="x" * 1000)]
    result = injector.format_injection(nodes)
    # Should be truncated to ~40 chars + tag overhead
    assert len(result) < 200


def test_custom_labels():
    injector = ContextInjector(labels={"themes": "Themen", "insights": "Erkenntnisse"})
    nodes = [
        make_node("performance", node_type="theme"),
        make_node("always use indexes", node_type="insight",
                  summary="Always add DB indexes for hot query paths"),
    ]
    result = injector.format_injection(nodes)
    assert "Themen" in result or "Erkenntnisse" in result


def test_event_nodes_formatted_with_header():
    injector = ContextInjector()
    nodes = [make_node("Deployment outage", node_type="event",
                        summary="Production went down for 2 hours")]
    result = injector.format_injection(nodes)
    assert "### Deployment outage" in result
