"""
Basic usage — BREATHE with in-memory stub backends.

Demonstrates the full pipeline without any external services.
Run with: python examples/basic_usage.py
"""
import asyncio
from breathe import (
    Synapse, GraphCompactor, BreatheConfig,
    MemoryRepository, LLMClient, RetrievedNode,
)
from breathe.config import ENGLISH


# --- Stub implementations ---

class StubMemoryRepository(MemoryRepository):
    """In-memory stub — returns hardcoded concepts and memories."""

    async def get_concepts(self) -> dict[str, str]:
        return {
            "FastAPI": "uuid-fastapi-001",
            "PostgreSQL": "uuid-postgres-001",
            "SYNAPSE": "uuid-synapse-001",
        }

    async def graph_bfs(self, start_ids, max_depth=2, min_strength=0.2, limit=20):
        return [
            RetrievedNode(
                node_id="uuid-related-001",
                concept="FastAPI async patterns",
                node_type="entity",
                summary="FastAPI handles async requests efficiently with async def handlers",
                importance=0.8,
                depth=1,
            )
        ]

    async def keyword_search(self, keywords, limit=5):
        return [
            RetrievedNode(
                node_id=f"mem-kw-{i}",
                concept=kw,
                node_type="memory_keyword",
                summary=f"Previous discussion about {kw}",
                importance=0.6,
                depth=0,
                memory_content=f"We discussed {kw} in detail last week and decided to use async handlers.",
            )
            for i, kw in enumerate(keywords[:2])
        ]


class StubLLMClient(LLMClient):
    """Returns hardcoded graph extraction for demo."""

    async def complete(self, prompt, max_tokens=4000, temperature=0.2):
        return """## Topics
- [API Design] [0.9] FastAPI endpoint architecture | PostgreSQL

## Decisions
- Use async handlers for all database operations

## Open
- Should we add Redis caching for heavy queries?

## Artifacts
- api/routes.py: main API router
"""


# --- Main demo ---

async def main():
    config = BreatheConfig(
        language_packs=[ENGLISH],
        default_language="en",
    )

    repo = StubMemoryRepository()
    synapse = Synapse(repository=repo, config=config, enable_model=False)
    await synapse.initialize()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How should I structure my FastAPI endpoints?"},
    ]

    print("--- Before injection ---")
    print(messages[-1]["content"])
    print()

    messages = await synapse.inject(messages)

    print("--- After SYNAPSE injection ---")
    print(messages[-1]["content"])
    print()

    # Show metrics
    stats = synapse.metrics.to_dict()
    print(f"SYNAPSE stats: {stats['synapse']['total_injections']} injections, "
          f"hit rate: {stats['synapse']['hit_rate']:.0%}")
    print()

    # Demonstrate GraphCompactor
    history = []
    for i in range(15):
        history.append({"role": "user", "content": f"User message {i}: detailed question about FastAPI, PostgreSQL, and async Python patterns. We discussed caching strategies, database connection pooling, and endpoint design."})
        history.append({"role": "assistant", "content": f"Assistant response {i}: Here is a detailed explanation covering async handlers, connection pool configuration, query optimization, and best practices for structuring FastAPI applications with PostgreSQL backends."})

    compactor = GraphCompactor(llm_client=StubLLMClient())
    result = await compactor.compress(history)
    print(f"GraphCompactor: compressed={result['compressed']}")
    if result["compressed"]:
        meta = result["metadata"]
        print(f"  {meta['original_tokens']} → {meta['compressed_tokens']} tokens "
              f"({meta['compression_ratio']:.0%} saved)")
        print(f"  Graph: {meta['graph_nodes']} nodes, {meta['graph_edges']} edges")


if __name__ == "__main__":
    asyncio.run(main())
