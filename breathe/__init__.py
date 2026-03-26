"""
BREATHE — context optimization for LLM applications.

Two-phase system:
  SYNAPSE (inhale)       — pre-generation memory injection
  GraphCompactor (exhale) — structured context compression

Quick start::

    from breathe import Synapse, GraphCompactor, BreatheConfig

    config = BreatheConfig()
    synapse = Synapse(repository=my_repo, config=config)
    await synapse.initialize()

    # Before each LLM call:
    messages = await synapse.inject(messages)

    # When context is getting full:
    compactor = GraphCompactor(llm_client=my_llm)
    result = await compactor.compress(messages)
    messages = result["compressed_messages"]
"""
from .config import BreatheConfig, LanguagePack, ENGLISH
from .interfaces import MemoryRepository, VectorSearchClient, LLMClient, RetrievedNode
from .anchor_extractor import AnchorExtractor, Anchor, AnchorResult
from .context_injector import ContextInjector
from .session_graph import SessionGraph, GraphNode, GraphEdge
from .graph_compactor import GraphCompactor
from .synapse import Synapse
from .metrics import BreatheMetrics

__version__ = "0.1.0"

__all__ = [
    # Core
    "Synapse",
    "GraphCompactor",
    "BreatheConfig",
    "LanguagePack",
    # Language packs
    "ENGLISH",
    # Interfaces
    "MemoryRepository",
    "VectorSearchClient",
    "LLMClient",
    "RetrievedNode",
    # Internals (for custom implementations)
    "AnchorExtractor",
    "Anchor",
    "AnchorResult",
    "ContextInjector",
    "SessionGraph",
    "GraphNode",
    "GraphEdge",
    "BreatheMetrics",
]
