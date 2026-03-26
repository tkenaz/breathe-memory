"""
Full integration — BREATHE + Memory Nexus with PostgreSQL.

Prerequisites:
    pip install breathe-memory[pg]
    createdb breathe_demo
    psql breathe_demo < schema.sql  # see memory_nexus/postgres.py for DDL

Run with: python examples/with_postgres.py
"""
import asyncio
import os
from breathe import Synapse, BreatheConfig
from breathe.config import ENGLISH
from memory_nexus import PostgresMemoryStore

DSN = os.environ.get("DATABASE_URL", "postgresql://localhost/breathe_demo")


async def main():
    # Initialize Memory Nexus store
    # Default: all-MiniLM-L6-v2 (384-dim, fast, good for prototyping)
    # Production: model_name="intfloat/multilingual-e5-large" (1024-dim)
    #   — remember to create your table with vector(1024) instead of vector(384)
    store = PostgresMemoryStore(
        dsn=DSN,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        min_similarity=0.55,
    )
    await store.initialize()

    # Store some memories
    await store.store(
        "FastAPI is a modern Python web framework for building APIs with automatic OpenAPI docs.",
        metadata=None,
    )
    await store.store(
        "PostgreSQL with pgvector extension enables vector similarity search in SQL.",
        metadata=None,
    )
    await store.store(
        "Redis is an in-memory data structure store used for caching and pub/sub.",
        metadata=None,
    )

    # Initialize SYNAPSE — store acts as both VectorSearchClient and can be
    # wrapped in a MemoryRepository adapter for graph BFS
    config = BreatheConfig(
        language_packs=[ENGLISH],
        default_language="en",
        min_similarity=0.55,
    )

    synapse = Synapse(
        vector_client=store,  # Memory Nexus as semantic search backend
        config=config,
        enable_model=False,
    )
    await synapse.initialize()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What should I use for caching in FastAPI?"},
    ]

    print("Query:", messages[-1]["content"])
    print()

    enriched = await synapse.inject(messages)

    print("Enriched message:")
    print(enriched[-1]["content"])
    print()

    # Search memories directly
    results = await store.search("caching database", limit=3)
    print(f"Direct search results ({len(results)} found):")
    for node in results:
        print(f"  [{node.importance:.2f}] {node.summary[:80]}...")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
