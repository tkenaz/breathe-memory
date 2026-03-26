# breathe-memory

**Context optimization and associative memory for LLM applications.**

Two-phase system built around how memory actually works — not as lookup, but as association.

```
pip install breathe-memory
```

---

## What it does

LLMs forget. Context windows are finite and expensive. Most solutions either stuff everything in (burns tokens) or summarize (loses structure).

**BREATHE** does neither:

- **SYNAPSE (inhale)** — before each generation, extracts associative anchors from the user message and injects semantically relevant memories directly into the prompt. The LLM starts thinking with context already loaded. Overhead: 2–20ms.

- **GraphCompactor (exhale)** — when context fills up, extracts a structured graph (topics, decisions, open questions, artifacts) instead of a lossy narrative summary. Typically saves 60–80% of tokens while preserving semantic structure.

```
                    ┌─────────────────────────────────────┐
    User message ──▶│           SYNAPSE (inhale)          │
                    │                                     │
                    │  1. Extract anchors (regex, 2ms)    │
                    │  2. Traverse memory graph (BFS)     │
                    │  3. Vector search (optional)        │
                    │  4. Inject <associative_memory>     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                              LLM with memory context
                                       │
                    ┌──────────────────▼──────────────────┐
                    │        GraphCompactor (exhale)       │
                    │     (fires when context ~80% full)   │
                    │                                     │
                    │  Compressible messages ──▶ LLM call │
                    │     → Topics, Decisions, Open,      │
                    │       Artifacts, Context, Dropped   │
                    │                                     │
                    │  Protected messages ──▶ kept intact │
                    └─────────────────────────────────────┘
```

---

## Quick start

```python
import asyncio
from breathe import Synapse, GraphCompactor, BreatheConfig
from breathe.interfaces import MemoryRepository, LLMClient, RetrievedNode

# Implement these two interfaces for your backend
class MyMemoryRepo(MemoryRepository):
    async def get_concepts(self):
        return {"FastAPI": "uuid-001", "Redis": "uuid-002"}

    async def graph_bfs(self, start_ids, **kwargs):
        return []  # implement BFS against your DB

    async def keyword_search(self, keywords, limit=5):
        return []  # implement ILIKE against your memories table

class MyLLMClient(LLMClient):
    async def complete(self, prompt, max_tokens=4000, temperature=0.2):
        # call your LLM API here
        ...

async def main():
    config = BreatheConfig()
    synapse = Synapse(repository=MyMemoryRepo(), config=config)
    await synapse.initialize()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How should I structure my FastAPI endpoints?"},
    ]

    # Inject associative memory before each LLM call
    messages = await synapse.inject(messages)

    # When context gets full, compress with GraphCompactor
    compactor = GraphCompactor(llm_client=MyLLMClient())
    result = await compactor.compress(messages)
    messages = result["compressed_messages"]

asyncio.run(main())
```

---

## With Memory Nexus (PostgreSQL + pgvector)

```python
from breathe import Synapse, BreatheConfig
from memory_nexus import PostgresMemoryStore

store = PostgresMemoryStore(dsn="postgresql://localhost/mydb")
await store.initialize()

# Store memories
await store.store("FastAPI handles async requests efficiently")
await store.store("Redis is ideal for session storage and caching")

# Wire into SYNAPSE — store implements VectorSearchClient
synapse = Synapse(vector_client=store, config=BreatheConfig())
await synapse.initialize()

messages = await synapse.inject(messages)
```

**PostgreSQL schema (default — 384-dim):**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE memories (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops);
```

**Embedding models:**

The default model (`all-MiniLM-L6-v2`, 384-dim, ~90 MB) is good for prototyping.
For production, we recommend `intfloat/multilingual-e5-large` (1024-dim, ~1.2 GB) — significantly better retrieval quality, especially for multilingual content.

To switch, pass `model_name` and adjust your table's vector dimension:

```python
store = PostgresMemoryStore(
    dsn="postgresql://localhost/mydb",
    model_name="intfloat/multilingual-e5-large",  # 1024-dim, multilingual
)
```
```sql
-- For e5-large, use vector(1024) instead of vector(384)
CREATE TABLE memories (
    ...
    embedding vector(1024),
    ...
);
```

---

## Language support

Built-in: **English**. Custom languages in ~10 lines:

```python
import re
from breathe import Synapse, BreatheConfig, LanguagePack

GERMAN = LanguagePack(
    code="de",
    stopwords=frozenset({"der", "die", "das", "und", "ist", ...}),
    hub_exclusions=frozenset({"system", "speicher"}),
    temporal_pattern=re.compile(r"\b(gestern|heute|morgen|neulich)\b", re.I),
    emotional_pattern=re.compile(r"\b(müde|glücklich|traurig|wütend)\b", re.I),
    labels={"themes": "Themen", "insights": "Erkenntnisse"},
)

config = BreatheConfig(language_packs=[GERMAN], default_language="de")
synapse = Synapse(config=config, ...)
```

Language packs control:
- **Stopwords** — excluded from relevance scoring
- **Hub exclusions** — nodes too generic to be useful for injection (e.g. "system", "memory"). Add your most frequent root concepts here — words that connect to everything are noise in retrieval. The more specific your exclusions, the sharper your injections.
- **Temporal and emotional regex patterns** — anchor extraction for time references and emotional signals
- **UI section labels** — headers used in the injected `<associative_memory>` block

---

## Architecture

### SYNAPSE pipeline (per-request, <200ms)

```
User message
     │
     ▼
AnchorExtractor
  ├─ Match known concepts (regex, 0.9 confidence)
  ├─ Temporal patterns   (0.7)
  ├─ Technical patterns  (0.5)
  └─ Emotional signals   (0.6)
     │
     ▼ [optional Phase 3 — Apple Silicon only]
ModelAnchorExtractor (local LLM via MLX, ~250ms)
  └─ Fires only when regex finds <5 matched nodes
     │
     ▼
Three traversal strategies (in parallel):
  1. Graph BFS    ── memory_nodes + memory_edges (recursive CTE)
  2. Vector search── any VectorSearchClient (pgvector, Pinecone, etc.)
  3. Keyword search── ILIKE on unmatched anchors
     │
     ▼
Relevance filter
  ├─ Hub exclusion (drop super-generic nodes)
  ├─ Session dedup (skip already-injected nodes)
  └─ Keyword overlap scoring (anchor words vs node content)
     │
     ▼
ContextInjector
  └─ <associative_memory> block → prepended to last user message
```

### GraphCompactor (when context fills up)

```
Old messages (compressible zone)
     │
     ▼ preprocess: compress tool call JSON
     ▼
LLM extraction call (your LLMClient)
     │
     ▼
SessionGraph: Topics / Decisions / Open / Artifacts / Context / Dropped
     │
     ▼
[SESSION GRAPH] message + protected recent messages
```

---

## Configuration

```python
from breathe import BreatheConfig
from breathe.config import ENGLISH

config = BreatheConfig(
    # Language packs (all active simultaneously)
    language_packs=[ENGLISH],
    default_language="en",

    # SYNAPSE tuning
    min_similarity=0.55,       # min vector similarity to accept
    max_injected_nodes=15,     # max nodes per injection
    enable_model_extractor=True,
    model_trigger_threshold=5, # model fires when regex finds <5 nodes

    # Token budgets by conversation mode
    mode_budgets={
        "casual":   1500,
        "work":     2500,
        "deep":     4000,
        "balanced": 2000,
    },

    # GraphCompactor
    compactor_model="claude-sonnet-4-20250514",
    compactor_fallback_model="claude-haiku-4-5-20251001",
    min_tokens_to_compress=300,
    protected_messages_normal=10,
)
```

---

## Implementing backends

### MemoryRepository (for graph BFS + keyword search)

```python
from breathe.interfaces import MemoryRepository, RetrievedNode

class MyRepo(MemoryRepository):
    async def get_concepts(self) -> dict[str, str]:
        # Return {concept_text: uuid} from your knowledge graph
        return {"Redis": "abc-123", "FastAPI": "def-456"}

    async def graph_bfs(self, start_ids, max_depth=2, **kwargs) -> list[RetrievedNode]:
        # BFS from start_ids through your concept graph
        # Recursive CTE on (memory_nodes, memory_edges) works well
        ...

    async def keyword_search(self, keywords, limit=5) -> list[RetrievedNode]:
        # ILIKE search over your memories/documents table
        ...

    async def flush_edges(self, edges) -> int:
        # Optional: persist new session graph edges to long-term storage
        return 0
```

### VectorSearchClient (for semantic search)

```python
from breathe.interfaces import VectorSearchClient, RetrievedNode

class PineconeClient(VectorSearchClient):
    async def search(self, query: str, limit: int = 5) -> list[RetrievedNode]:
        # embed query, search your vector index, return RetrievedNode list
        ...
```

### LLMClient (for GraphCompactor)

```python
from breathe.interfaces import LLMClient

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str):
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt, max_tokens=4000, temperature=0.2):
        msg = await self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

class OpenAIClient(LLMClient):
    async def complete(self, prompt, max_tokens=4000, temperature=0.2):
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
```

---

## Performance

Measured in production on Apple M2 Max:

| Component              | Latency     | Notes |
|------------------------|-------------|-------|
| Regex extraction       | 2ms         | always runs |
| MLX model extraction   | ~250ms      | conditional (when regex < 5 matches) |
| Graph BFS (PG)         | 5–15ms      | recursive CTE, depth=2 |
| Vector search (pgvector)| 10–30ms    | depends on index size |
| Keyword search (ILIKE) | 3–10ms      | depends on table size |
| **Total SYNAPSE**      | **2–60ms**  | without model |
| **Total SYNAPSE**      | **~300ms**  | with model |
| GraphCompactor         | 3–8s        | one LLM call, happens rarely |

GraphCompactor fires infrequently (only at ~80% context fill), so its latency
doesn't affect per-request response time.

---

## Memory management

BREATHE handles **retrieval and injection** automatically. **Storing memories is your application's responsibility** — you decide what to remember and when.

```python
# Your application stores memories explicitly
await store.store("User prefers dark mode and concise answers")
await store.store("Project uses FastAPI + PostgreSQL + Redis stack")

# SYNAPSE retrieves relevant ones automatically before each LLM call
messages = await synapse.inject(messages)
```

This is intentional: memory storage policies (what to keep, when to forget, privacy rules) vary wildly between applications. BREATHE gives you the retrieval engine — you control the data.

> **Coming soon:** A standalone MCP server wrapping Memory Nexus, so LLMs can store and search memories directly as tool calls.

---

## Optional dependencies

```bash
# PostgreSQL + pgvector backend
pip install breathe-memory[pg]

# Apple Silicon local model extractor (MLX)
pip install breathe-memory[mlx]

# Anthropic client for GraphCompactor
pip install breathe-memory[anthropic]

# OpenAI client for GraphCompactor
pip install breathe-memory[openai]

# Everything
pip install breathe-memory[all]
```

Core package has zero dependencies beyond Python stdlib + `typing-extensions`.

### Model extractor (Phase 3)

The optional `ModelAnchorExtractor` uses [MLX](https://github.com/ml-explore/mlx) to run a small local LLM for contextual anchor extraction when regex alone isn't enough.

**This requires Apple Silicon (M1/M2/M3/M4).** MLX is an Apple-only framework and will not work on Linux or Windows. If MLX is not installed, the model extractor is silently skipped — everything else works normally.

The default model is `Qwen3-1.7B` (4-bit, ~1.2 GB RAM). You can swap it for any MLX-compatible model by passing `model_id` to `ModelAnchorExtractor`. If you need cross-platform model extraction, implement your own extractor using any inference backend (ollama, vLLM, API calls) — the interface is a single `extract(message) -> list[Anchor]` method.

---

## Monitoring

```python
from breathe import BreatheMetrics

stats = BreatheMetrics.get().to_dict()
# {
#   "synapse": {
#     "total_injections": 142,
#     "hit_rate": 0.87,
#     "latency": {"avg_ms": 18.3, "p95_ms": 45.1},
#     "top_anchors": [{"text": "FastAPI", "count": 23}, ...]
#   },
#   "compaction": {
#     "total": 3,
#     "avg_ratio": 0.71,
#     "total_saved_tokens": 12400
#   }
# }
```

Expose via your API: `GET /api/breathe-stats` → `BreatheMetrics.get().to_dict()`

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Built by [Kenaz GmbH](https://kenaz.ai) — Custom AI Agents, MCP Servers, Semantic Engineering.
