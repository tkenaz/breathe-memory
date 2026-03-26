"""
Memory Nexus — hybrid semantic memory for LLM applications.

Provides persistent memory storage with semantic search via pgvector.
Integrates with BREATHE as a VectorSearchClient backend.

Architecture:
  MemoryStore   — abstract storage interface
  PostgresStore — asyncpg + pgvector implementation (optional dep)
  HybridSearch  — dense + keyword search

Quick start::

    from memory_nexus import PostgresMemoryStore
    from breathe import Synapse

    store = PostgresMemoryStore(dsn="postgresql://...")
    await store.initialize()

    synapse = Synapse(vector_client=store)
"""
from .models import Memory, MemoryMetadata
from .store import MemoryStore

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "MemoryMetadata",
    "MemoryStore",
]

# Optional: PostgreSQL backend (requires asyncpg + pgvector)
try:
    from .postgres import PostgresMemoryStore
    __all__.append("PostgresMemoryStore")
except ImportError:
    pass
