"""
Interfaces — abstract contracts for external dependencies.

BREATHE is storage-agnostic and LLM-agnostic by design.
Implement these interfaces to integrate with any backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievedNode:
    """A node retrieved from graph traversal or memory search."""

    node_id: str
    concept: str
    node_type: str  # entity, theme, insight, event, memory_vector, memory_keyword
    summary: Optional[str] = None
    importance: float = 0.5
    depth: int = 0  # BFS depth from anchor
    relation: Optional[str] = None  # edge relation that led here
    memory_content: Optional[str] = None  # raw text from memory store


class MemoryRepository(ABC):
    """
    Abstract storage backend for BREATHE.

    Implement this to connect BREATHE to your own database.
    See ``breathe.backends.postgres`` for the asyncpg/PostgreSQL reference
    implementation.

    The interface deliberately stays minimal — you only need what SYNAPSE
    actually calls. Graph BFS and keyword search are optional; implement the
    ones you need and raise ``NotImplementedError`` for the rest.
    """

    @abstractmethod
    async def get_concepts(self) -> dict[str, str]:
        """
        Return all active known concepts as ``{concept_text: node_uuid}``.

        Called once at initialization to build the concept regex.
        An empty dict is valid — SYNAPSE falls back to regex-only extraction.
        """

    @abstractmethod
    async def graph_bfs(
        self,
        start_ids: list[str],
        max_depth: int = 2,
        min_strength: float = 0.2,
        limit: int = 20,
    ) -> list[RetrievedNode]:
        """
        BFS traversal from ``start_ids`` through the concept graph.

        Args:
            start_ids: UUIDs of matched memory_nodes.
            max_depth: Maximum edge hops to follow.
            min_strength: Minimum edge strength to follow (0–1).
            limit: Maximum nodes to return.

        Returns:
            List of RetrievedNode, sorted by importance descending.
        """

    @abstractmethod
    async def keyword_search(
        self, keywords: list[str], limit: int = 5
    ) -> list[RetrievedNode]:
        """
        Full-text keyword search over memory content.

        Called for anchors that didn't match any known concept (no node_id).
        ILIKE or equivalent is fine — precision matters less than recall here.

        Args:
            keywords: Words to search for (case-insensitive).
            limit: Maximum memories to return.
        """

    async def flush_edges(self, edges: list) -> int:
        """
        Persist new session graph edges to long-term storage.

        Called at session end by SessionGraph.flush(). Optional — if you don't
        need cross-session graph persistence, leave this as a no-op.

        Returns number of edges flushed.
        """
        return 0


class VectorSearchClient(ABC):
    """
    Abstract client for semantic / vector search.

    Wraps any dense-embedding search backend (pgvector, Pinecone, Weaviate, etc.).
    BREATHE uses this for Strategy 2 in SYNAPSE traversal — the highest-quality
    but most expensive retrieval path.
    """

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[RetrievedNode]:
        """
        Return the most semantically similar memories for ``query``.

        Args:
            query: Short anchor phrase (NOT the full user message).
            limit: Max results.

        Returns:
            List of RetrievedNode sorted by similarity descending.
            Set ``importance`` to the similarity score (0–1).
        """


class LLMClient(ABC):
    """
    Abstract LLM client for GraphCompactor.

    GraphCompactor needs a single call: given a long prompt, return text.
    Implement this to use any LLM (Anthropic, OpenAI, local, etc.).
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> Optional[str]:
        """
        Generate a completion for ``prompt``.

        Args:
            prompt: The full extraction prompt (can be long).
            max_tokens: Max tokens to generate.
            temperature: Low values (0.1–0.3) work best for structured extraction.

        Returns:
            Generated text, or None on failure.
        """
