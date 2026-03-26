"""Tests for SessionGraph."""
import pytest
from breathe.session_graph import SessionGraph, GraphNode, GraphEdge


def make_graph():
    g = SessionGraph(session_id="test-session")
    a = g.add_node(GraphNode(id="a", node_type="topic", label="API", content="REST API design"))
    b = g.add_node(GraphNode(id="b", node_type="decision", label="Use FastAPI", content="Chose FastAPI for async support"))
    c = g.add_node(GraphNode(id="c", node_type="artifact", label="main.py", content="Application entry point"))
    g.add_edge("a", "b", edge_type="led_to", weight=0.8)
    g.add_edge("b", "c", edge_type="part_of", weight=0.7)
    return g


def test_add_node():
    g = SessionGraph()
    node = g.add_node(GraphNode(id="x", node_type="topic", label="Test", content="test content"))
    assert g.node_count == 1
    assert g.get_node("x") is not None


def test_add_node_activates_existing():
    g = SessionGraph()
    g.add_node(GraphNode(id="x", node_type="topic", label="Test", content="content", weight=0.5))
    g.add_node(GraphNode(id="x", node_type="topic", label="Test", content="content"))
    assert g.node_count == 1  # no duplicate
    assert g.get_node("x").weight > 0.5  # activated


def test_add_edge():
    g = make_graph()
    assert g.edge_count == 2


def test_add_edge_missing_node():
    g = SessionGraph()
    g.add_node(GraphNode(id="a", node_type="topic", label="A", content="content A"))
    result = g.add_edge("a", "nonexistent")
    assert result is None


def test_traverse_bfs():
    g = make_graph()
    results = g.traverse(["a"], max_depth=2)
    node_ids = [n.id for n, _ in results]
    assert "a" in node_ids
    assert "b" in node_ids  # depth 1
    assert "c" in node_ids  # depth 2


def test_traverse_max_depth():
    g = make_graph()
    results = g.traverse(["a"], max_depth=1)
    node_ids = [n.id for n, _ in results]
    assert "a" in node_ids
    assert "b" in node_ids
    assert "c" not in node_ids  # too deep


def test_traverse_sorted_by_weight():
    g = SessionGraph()
    g.add_node(GraphNode(id="low", node_type="topic", label="low", content="", weight=0.3))
    g.add_node(GraphNode(id="high", node_type="topic", label="high", content="", weight=0.9))
    results = g.traverse(["low", "high"])
    assert results[0][0].id == "high"


def test_remove_node():
    g = make_graph()
    g.remove_node("b")
    assert g.node_count == 2
    assert g.get_node("b") is None


def test_serialization_roundtrip():
    g = make_graph()
    text = g.to_structured_text()
    assert "## Topics" in text
    assert "## Decisions" in text

    restored = SessionGraph.from_structured_text(text)
    assert restored.node_count > 0


def test_empty_graph_serialization():
    g = SessionGraph()
    assert g.to_structured_text() == ""


def test_repr():
    g = make_graph()
    r = repr(g)
    assert "nodes=3" in r
    assert "edges=2" in r
