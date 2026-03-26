"""
SessionGraph — in-memory graph for the current session's warm layer.

Sits between hot context (full message history) and cold storage (vector DB).
Built during compaction (Graph Compactor), traversed by SYNAPSE, flushed to
persistent storage at session end.

Design principle: the LLM decides what to record. No mechanical fixation.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the session graph — topic, decision, artifact, or state."""

    id: str
    node_type: str  # 'topic', 'decision', 'artifact', 'open_question', 'state'
    label: str      # short display name
    content: str    # full content / description
    weight: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activated: datetime = field(default_factory=datetime.utcnow)
    source_memory_id: Optional[str] = None

    def activate(self, boost: float = 0.1) -> None:
        """Strengthen node when referenced again."""
        self.weight = min(1.0, self.weight + boost)
        self.last_activated = datetime.utcnow()

    def decay(self, factor: float = 0.95, floor: float = 0.1) -> None:
        """Exponential decay with floor. Nothing fully disappears."""
        self.weight = max(floor, self.weight * factor)


@dataclass
class GraphEdge:
    """Directed edge between nodes."""

    from_id: str
    to_id: str
    edge_type: str  # 'relates_to', 'led_to', 'part_of', 'temporal_parallel', 'decided'
    weight: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None


class SessionGraph:
    """
    In-memory graph for current session.

    Warm layer between hot (full text) and cold (vector DB).
    Built during compaction, traversed by SYNAPSE, flushed at session end.

    Flush to persistent storage is optional — pass a ``MemoryRepository``
    to ``flush()`` if you want cross-session graph continuity.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, list[GraphEdge]] = {}  # from_id → [edges]
        self._new_edges: list[GraphEdge] = []
        self._new_nodes: list[GraphNode] = []

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.edges.values())

    # --- Node operations ---

    def add_node(self, node: GraphNode) -> GraphNode:
        """Add a node. If it already exists, activate (strengthen) it."""
        if node.id in self.nodes:
            existing = self.nodes[node.id]
            existing.activate()
            if len(node.content) > len(existing.content):
                existing.content = node.content
            return existing
        self.nodes[node.id] = node
        self._new_nodes.append(node)
        return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        self.edges.pop(node_id, None)
        for from_id in list(self.edges.keys()):
            self.edges[from_id] = [e for e in self.edges[from_id] if e.to_id != node_id]
        return True

    # --- Edge operations ---

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str = "relates_to",
        weight: float = 0.5,
    ) -> Optional[GraphEdge]:
        """Connect two nodes. Strengthens existing edge if already present."""
        if from_id not in self.nodes or to_id not in self.nodes:
            logger.warning(f"Cannot create edge: node(s) missing ({from_id} → {to_id})")
            return None
        for edge in self.edges.get(from_id, []):
            if edge.to_id == to_id and edge.edge_type == edge_type:
                edge.weight = min(1.0, edge.weight + 0.1)
                return edge
        edge = GraphEdge(
            from_id=from_id,
            to_id=to_id,
            edge_type=edge_type,
            weight=weight,
            session_id=self.session_id,
        )
        self.edges.setdefault(from_id, []).append(edge)
        self._new_edges.append(edge)
        return edge

    def drop_edge(self, from_id: str, to_id: str, edge_type: Optional[str] = None) -> bool:
        if from_id not in self.edges:
            return False
        before = len(self.edges[from_id])
        if edge_type:
            self.edges[from_id] = [
                e for e in self.edges[from_id]
                if not (e.to_id == to_id and e.edge_type == edge_type)
            ]
        else:
            self.edges[from_id] = [e for e in self.edges[from_id] if e.to_id != to_id]
        return len(self.edges[from_id]) < before

    # --- Traversal ---

    def traverse(
        self,
        start_ids: list[str],
        max_depth: int = 3,
        min_weight: float = 0.3,
        max_nodes: int = 20,
    ) -> list[tuple[GraphNode, int]]:
        """
        BFS traversal from start nodes.

        Returns ``(node, depth)`` pairs sorted by weight descending.
        In-memory: <1ms for typical session graphs.
        """
        visited: set[str] = set()
        result: list[tuple[GraphNode, int]] = []
        queue: deque[tuple[str, int]] = deque()

        for start_id in start_ids:
            if start_id in self.nodes:
                queue.append((start_id, 0))
                visited.add(start_id)

        while queue and len(result) < max_nodes:
            node_id, depth = queue.popleft()
            node = self.nodes.get(node_id)
            if node:
                result.append((node, depth))
            if depth < max_depth:
                for edge in self.edges.get(node_id, []):
                    if edge.to_id not in visited and edge.weight >= min_weight:
                        visited.add(edge.to_id)
                        queue.append((edge.to_id, depth + 1))

        result.sort(key=lambda x: x[0].weight, reverse=True)
        return result

    # --- Serialization ---

    def to_structured_text(self, max_tokens: int = 10000) -> str:
        """
        Serialize graph to structured text for context window.

        This is the warm-layer representation — what the LLM sees instead
        of a lossy summary. Format matches the GraphCompactor extraction prompt
        so it can be round-tripped through from_structured_text().
        """
        if not self.nodes:
            return ""

        sections: list[str] = []
        by_type: dict[str, list[GraphNode]] = {}
        for node in sorted(self.nodes.values(), key=lambda n: n.weight, reverse=True):
            by_type.setdefault(node.node_type, []).append(node)

        if "topic" in by_type:
            lines = ["## Topics"]
            for node in by_type["topic"]:
                connected = []
                for edge in self.edges.get(node.id, []):
                    target = self.nodes.get(edge.to_id)
                    if target:
                        connected.append(f"{target.label} ({edge.edge_type})")
                conn_str = f" | {', '.join(connected)}" if connected else ""
                lines.append(f"- [{node.label}] [{node.weight:.1f}] {node.content}{conn_str}")
            sections.append("\n".join(lines))

        if "decision" in by_type:
            lines = ["## Decisions"]
            for node in by_type["decision"]:
                lines.append(f"- {node.content}")
            sections.append("\n".join(lines))

        if "open_question" in by_type:
            lines = ["## Open"]
            for node in by_type["open_question"]:
                lines.append(f"- {node.content}")
            sections.append("\n".join(lines))

        if "artifact" in by_type:
            lines = ["## Artifacts"]
            for node in by_type["artifact"]:
                lines.append(f"- {node.label}: {node.content}")
            sections.append("\n".join(lines))

        if "state" in by_type:
            lines = ["## Context"]
            for node in by_type["state"]:
                lines.append(f"- {node.content}")
            sections.append("\n".join(lines))

        text = "\n\n".join(sections)
        char_budget = max_tokens * 4
        if len(text) > char_budget:
            text = text[:char_budget] + "\n[... graph truncated to fit budget]"
        return text

    @classmethod
    def from_structured_text(
        cls, text: str, session_id: Optional[str] = None
    ) -> "SessionGraph":
        """
        Parse structured text back into a SessionGraph.

        Best-effort parser — used when reloading a previous compaction result
        from the context window. The graph is the source of truth; the text
        is a serialization.
        """
        graph = cls(session_id=session_id)
        if not text.strip():
            return graph

        current_type = "topic"
        type_map = {
            "## Topics": "topic",
            "## Decisions": "decision",
            "## Open": "open_question",
            "## Artifacts": "artifact",
            "## Context": "state",
        }
        node_counter = 0

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line in type_map:
                current_type = type_map[line]
                continue
            if line.startswith("- "):
                content = line[2:].strip()
                node_counter += 1
                node_id = f"restored_{node_counter}"

                label = content
                if content.startswith("["):
                    bracket_end = content.find("]")
                    if bracket_end > 0:
                        label = content[1:bracket_end]
                        content = content[bracket_end + 1:].strip()

                weight = 0.5
                if content.startswith("[") and "]" in content:
                    weight_end = content.find("]")
                    try:
                        weight = float(content[1:weight_end])
                    except ValueError:
                        pass
                    content = content[weight_end + 1:].strip()

                graph.add_node(GraphNode(
                    id=node_id,
                    node_type=current_type,
                    label=label,
                    content=content or label,
                    weight=weight,
                ))

        return graph

    # --- Persistence ---

    async def flush(self, repository=None) -> int:
        """
        Persist new edges and nodes to long-term storage.

        Args:
            repository: A ``MemoryRepository`` instance. If None, no-op.

        Returns:
            Number of items flushed.
        """
        if repository is None or (not self._new_edges and not self._new_nodes):
            return 0
        flushed = await repository.flush_edges(self._new_edges)
        self._new_edges.clear()
        self._new_nodes.clear()
        return flushed

    def __repr__(self) -> str:
        return (
            f"SessionGraph(nodes={self.node_count}, "
            f"edges={self.edge_count}, session={self.session_id})"
        )
