"""
BREATHE Metrics — in-memory analytics for SYNAPSE + GraphCompactor.

Singleton collector. Thread-safe via simple dict/list operations.
No external dependencies.

Expose via your API to power a real-time monitoring dashboard.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MAX_HISTORY = 200


@dataclass
class AnchorDetail:
    text: str
    anchor_type: str  # entity, temporal, technical, emotional
    confidence: float
    source: str  # regex, known_concept, model
    matched_node_id: Optional[str] = None


@dataclass
class NodeDetail:
    concept: str
    node_type: str
    summary: Optional[str] = None
    importance: float = 0.5
    depth: int = 0
    relation: Optional[str] = None


@dataclass
class SynapseEvent:
    timestamp: float
    anchors_count: int
    matched_nodes: int
    injected_nodes: int
    regex_ms: float
    model_ms: float = 0.0
    total_ms: float = 0.0
    mode: str = "balanced"
    model_triggered: bool = False
    user_message: str = ""
    anchors: list[AnchorDetail] = field(default_factory=list)
    nodes: list[NodeDetail] = field(default_factory=list)
    injection_text: str = ""


@dataclass
class CompactionEvent:
    timestamp: float
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    compression_ratio: float
    graph_nodes: int
    graph_edges: int
    strategy: str = "graph"
    fallback_used: bool = False
    duration_ms: float = 0.0
    extracted_text: str = ""


class BreatheMetrics:
    """
    In-memory metrics collector for BREATHE.

    Singleton pattern — one instance shared across Synapse and GraphCompactor.
    Access via ``BreatheMetrics.get()``.

    Serialize to dict for API exposure via ``to_dict()``.
    """

    _instance: Optional["BreatheMetrics"] = None

    def __init__(self, history_size: int = MAX_HISTORY):
        self._history_size = history_size

        # SYNAPSE
        self.synapse_events: list[SynapseEvent] = []
        self.synapse_total: int = 0
        self.synapse_skipped: int = 0

        # Compaction
        self.compaction_events: list[CompactionEvent] = []
        self.compaction_total: int = 0
        self.compaction_fallbacks: int = 0

        # State
        self.known_concepts_count: int = 0
        self.model_triggers: int = 0
        self.model_available: bool = False

        # Graph flush
        self.flush_count: int = 0
        self.flush_edges_total: int = 0

        # Anchor frequency
        self._anchor_counts: dict[str, int] = {}

        self.started_at: float = time.time()

    @classmethod
    def get(cls) -> "BreatheMetrics":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful for testing)."""
        cls._instance = None

    def record_synapse(self, event: SynapseEvent) -> None:
        self.synapse_events.append(event)
        if len(self.synapse_events) > self._history_size:
            self.synapse_events = self.synapse_events[-self._history_size:]
        self.synapse_total += 1

    def record_synapse_skip(self) -> None:
        self.synapse_skipped += 1

    def record_anchors(self, anchor_texts: list[str]) -> None:
        for text in anchor_texts:
            key = text.lower()
            self._anchor_counts[key] = self._anchor_counts.get(key, 0) + 1

    def record_compaction(self, event: CompactionEvent) -> None:
        self.compaction_events.append(event)
        if len(self.compaction_events) > self._history_size:
            self.compaction_events = self.compaction_events[-self._history_size:]
        self.compaction_total += 1
        if event.fallback_used:
            self.compaction_fallbacks += 1

    def record_flush(self, edges_count: int) -> None:
        self.flush_count += 1
        self.flush_edges_total += edges_count

    def to_dict(self) -> dict:
        """Export metrics as JSON-serializable dict."""
        now = time.time()
        uptime_s = now - self.started_at

        synapse_latencies = [e.total_ms for e in self.synapse_events]
        regex_latencies = [e.regex_ms for e in self.synapse_events]
        model_latencies = [e.model_ms for e in self.synapse_events if e.model_triggered]
        comp_ratios = [e.compression_ratio for e in self.compaction_events]
        comp_saved = [e.saved_tokens for e in self.compaction_events]

        total = self.synapse_total + self.synapse_skipped
        hit_rate = self.synapse_total / total if total > 0 else 0.0

        top_anchors = sorted(
            self._anchor_counts.items(), key=lambda x: x[1], reverse=True
        )[:15]

        recent_synapse = [
            {
                "timestamp": e.timestamp,
                "anchors": e.anchors_count,
                "nodes": e.injected_nodes,
                "total_ms": round(e.total_ms, 1),
                "mode": e.mode,
                "model_used": e.model_triggered,
                "user_message": e.user_message,
                "anchors_detail": [
                    {
                        "text": a.text, "type": a.anchor_type,
                        "confidence": a.confidence, "source": a.source,
                        "matched": a.matched_node_id is not None,
                    }
                    for a in e.anchors
                ],
                "nodes_detail": [
                    {
                        "concept": n.concept, "type": n.node_type,
                        "summary": n.summary, "importance": n.importance,
                        "depth": n.depth, "relation": n.relation,
                    }
                    for n in e.nodes
                ],
                "injection_text": e.injection_text,
            }
            for e in self.synapse_events[-20:]
        ]

        recent_compactions = [
            {
                "timestamp": e.timestamp,
                "original_tokens": e.original_tokens,
                "compressed_tokens": e.compressed_tokens,
                "ratio": round(e.compression_ratio, 2),
                "nodes": e.graph_nodes,
                "edges": e.graph_edges,
                "fallback": e.fallback_used,
                "duration_ms": round(e.duration_ms, 1),
                "extracted_text": e.extracted_text,
            }
            for e in self.compaction_events[-10:]
        ]

        return {
            "uptime_seconds": round(uptime_s),
            "synapse": {
                "total_injections": self.synapse_total,
                "total_skipped": self.synapse_skipped,
                "hit_rate": round(hit_rate, 3),
                "known_concepts": self.known_concepts_count,
                "model_available": self.model_available,
                "model_triggers": self.model_triggers,
                "latency": {
                    "avg_ms": round(_avg(synapse_latencies), 1),
                    "p50_ms": round(_percentile(synapse_latencies, 0.5), 1),
                    "p95_ms": round(_percentile(synapse_latencies, 0.95), 1),
                    "max_ms": round(max(synapse_latencies), 1) if synapse_latencies else 0,
                    "regex_avg_ms": round(_avg(regex_latencies), 1),
                    "model_avg_ms": round(_avg(model_latencies), 1) if model_latencies else 0,
                },
                "recent": recent_synapse,
            },
            "compaction": {
                "total": self.compaction_total,
                "fallbacks": self.compaction_fallbacks,
                "avg_ratio": round(_avg(comp_ratios), 2) if comp_ratios else 0,
                "total_saved_tokens": sum(comp_saved),
                "recent": recent_compactions,
            },
            "graph": {
                "flush_count": self.flush_count,
                "flush_edges_total": self.flush_edges_total,
                "top_anchors": [
                    {"text": text, "count": count} for text, count in top_anchors
                ],
            },
        }


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]
