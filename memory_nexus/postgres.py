"""
PostgreSQL + pgvector implementation of MemoryStore.

Requires:
    pip install breathe-memory[pg]

Database setup (default — 384-dim model):
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE memories (
        id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
        content TEXT NOT NULL,
        embedding vector(384),
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops);
    CREATE INDEX ON memories (created_at DESC);

Embedding models:
    Default: sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~90 MB, English).
    Good for prototyping and quick starts.

    Production: intfloat/multilingual-e5-large (1024-dim, ~1.2 GB, multilingual).
    Significantly better retrieval quality, especially for non-English text.
    To use: pass model_name="intfloat/multilingual-e5-large" and create
    your table with ``embedding vector(1024)`` instead of ``vector(384)``.

    The vector dimension is determined by the model — make sure your
    table column matches the model output dimension.
"""
from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from .models import Memory, MemoryMetadata
from .store import MemoryStore
from breathe.interfaces import RetrievedNode

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="breathe-embed")


def _mps_available() -> bool:
    """True when sentence-transformers would place the model on Apple's MPS backend."""
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def load_embedding_model(model_name: str):
    """
    Load a sentence-transformer so that short texts embed correctly on every device.

    On Apple's MPS backend, PyTorch's scaled-dot-product attention kernel returns
    wrong hidden states for XLM-RoBERTa-based models (multilingual-e5, bge-m3) when
    a short sequence is encoded on its own; the same text inside a longer batch is
    correct, which hides the problem. Eager attention is exact on MPS, so it is
    requested whenever MPS is available. Models that reject the option are loaded
    on CPU instead: slower, but correct.
    """
    from sentence_transformers import SentenceTransformer

    if not _mps_available():
        return SentenceTransformer(model_name)
    try:
        return SentenceTransformer(
            model_name, model_kwargs={"attn_implementation": "eager"}
        )
    except (TypeError, ValueError) as e:
        logger.warning(
            f"{model_name} does not accept eager attention ({e}); loading on CPU"
        )
        return SentenceTransformer(model_name, device="cpu")


class PostgresMemoryStore(MemoryStore):
    """
    asyncpg + pgvector memory store with sentence-transformer embeddings.

    Usage::

        store = PostgresMemoryStore(
            dsn="postgresql://user:pass@localhost/mydb",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        await store.initialize()

        mem = await store.store("Claude is an AI assistant made by Anthropic")
        results = await store.search("AI assistant", limit=3)
    """

    def __init__(
        self,
        dsn: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_similarity: float = 0.55,
    ):
        """
        Args:
            dsn: PostgreSQL connection string.
            model_name: Sentence-transformer model for embeddings.
                        Default: all-MiniLM-L6-v2 (384-dim, fast, good for prototyping).
                        Production: intfloat/multilingual-e5-large (1024-dim, multilingual).
                        Make sure your ``embedding vector(N)`` column matches the model dimension.
            min_similarity: Minimum cosine similarity to return from search.
        """
        self._dsn = dsn
        self._model_name = model_name
        self._min_similarity = min_similarity
        self._pool = None
        self._model = None

    async def initialize(self) -> None:
        """Create connection pool and load embedding model."""
        try:
            import asyncpg
        except ImportError as e:
            raise ImportError(
                "asyncpg is required for PostgresMemoryStore. "
                "Install with: pip install breathe-memory[pg]"
            ) from e

        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)

        # Register pgvector codec
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.set_type_codec(
                "vector",
                encoder=lambda v: "[" + ",".join(map(str, v)) + "]",
                decoder=lambda v: list(map(float, v.strip("[]").split(","))),
                schema="pg_catalog",
                format="text",
            )

        logger.info(f"PostgresMemoryStore initialized (model: {self._model_name})")

    def _get_model(self):
        """Lazy-load embedding model."""
        if self._model is None:
            try:
                self._model = load_embedding_model(self._model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install breathe-memory[pg]"
                ) from e
        return self._model

    async def _embed(self, text: str) -> list[float]:
        """Encode text to embedding vector (runs in thread pool)."""
        import asyncio
        loop = asyncio.get_running_loop()
        model = self._get_model()
        vector = await loop.run_in_executor(
            _executor, lambda: model.encode(text).tolist()
        )
        return vector

    async def store(
        self,
        content: str,
        metadata: Optional[MemoryMetadata] = None,
    ) -> Memory:
        if not self._pool:
            raise RuntimeError("Call initialize() first")

        meta = metadata or MemoryMetadata()
        embedding = await self._embed(content)
        memory_id = str(uuid.uuid4())

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (id, content, embedding, metadata, created_at)
                VALUES ($1, $2, $3::vector, $4, NOW())
                RETURNING id, created_at
                """,
                memory_id,
                content,
                embedding,
                json.dumps(meta.to_dict()),
            )

        return Memory(
            id=row["id"],
            content=content,
            metadata=meta,
            created_at=row["created_at"],
        )

    async def get(self, memory_id: str) -> Optional[Memory]:
        if not self._pool:
            raise RuntimeError("Call initialize() first")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, content, metadata, created_at FROM memories WHERE id = $1",
                memory_id,
            )
        if not row:
            return None

        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        return Memory(
            id=row["id"],
            content=row["content"],
            metadata=MemoryMetadata.from_dict(meta or {}),
            created_at=row["created_at"],
        )

    async def delete(self, memory_id: str) -> bool:
        if not self._pool:
            raise RuntimeError("Call initialize() first")
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1", memory_id
            )
        return result.split()[-1] == "1"

    async def search(self, query: str, limit: int = 5) -> list[RetrievedNode]:
        if not self._pool:
            raise RuntimeError("Call initialize() first")

        embedding = await self._embed(query)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM memories
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding,
                self._min_similarity,
                limit,
            )

        nodes = []
        for row in rows:
            content = row["content"] or ""
            meta = row["metadata"] or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            similarity = float(row["similarity"])
            tags = meta.get("tags", [])
            concept = tags[0] if tags else content[:50]
            nodes.append(RetrievedNode(
                node_id=row["id"],
                concept=concept,
                node_type="memory_vector",
                summary=content[:500] if len(content) > 500 else content,
                importance=similarity,
                depth=0,
                memory_content=content,
            ))

        return nodes

    async def get_recent(self, limit: int = 20) -> list[Memory]:
        if not self._pool:
            raise RuntimeError("Call initialize() first")

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata, created_at
                FROM memories
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

        memories = []
        for row in rows:
            meta = row["metadata"] or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            memories.append(Memory(
                id=row["id"],
                content=row["content"],
                metadata=MemoryMetadata.from_dict(meta),
                created_at=row["created_at"],
            ))
        return memories

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
