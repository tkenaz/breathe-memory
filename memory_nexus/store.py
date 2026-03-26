"""Abstract MemoryStore interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .models import Memory, MemoryMetadata
from breathe.interfaces import RetrievedNode, VectorSearchClient


class MemoryStore(VectorSearchClient, ABC):
    """
    Abstract memory store — combines VectorSearchClient with CRUD operations.

    Implements VectorSearchClient so it can be passed directly to Synapse::

        store = PostgresMemoryStore(dsn="postgresql://...")
        synapse = Synapse(vector_client=store)

    Implement this to connect Memory Nexus to any backend.
    See ``PostgresMemoryStore`` for the asyncpg + pgvector reference implementation.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the store (create tables, load models, etc.)."""

    @abstractmethod
    async def store(
        self,
        content: str,
        metadata: Optional[MemoryMetadata] = None,
    ) -> Memory:
        """
        Store a memory and return it with generated ID.

        Args:
            content: The text content to store.
            metadata: Optional metadata (tags, source, importance).
        """

    @abstractmethod
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[RetrievedNode]:
        """
        Semantic search. Implements VectorSearchClient.

        Args:
            query: Search query text.
            limit: Maximum results.

        Returns:
            List of RetrievedNode sorted by similarity descending.
        """

    @abstractmethod
    async def get_recent(self, limit: int = 20) -> list[Memory]:
        """Return most recently stored memories."""

    async def close(self) -> None:
        """Clean up resources (close connections, etc.)."""
