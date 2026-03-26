"""
SYNAPSE — pre-generation associative memory injection.

The inhale of BREATHE. Fires BEFORE generation, not after tool call.
This is middleware, not a tool — because tools are reactive (called),
SYNAPSE must be proactive (inject before thinking).

Pipeline:
  [User message] → anchor extraction → graph traversal → relevance filter → injection
  Total overhead target: <200ms

Usage::

    synapse = Synapse(repository=my_repo, config=BreatheConfig())
    await synapse.initialize()
    messages = await synapse.inject(messages)
    # pass enriched messages to your LLM
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from .anchor_extractor import AnchorExtractor, AnchorResult, Anchor
from .config import BreatheConfig
from .context_injector import ContextInjector
from .interfaces import MemoryRepository, VectorSearchClient, RetrievedNode
from .metrics import BreatheMetrics, SynapseEvent, AnchorDetail, NodeDetail

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[\w\u0400-\u04FF]{3,}", re.UNICODE)


class Synapse:
    """
    Pre-generation memory injection middleware.

    Intercepts messages before they go to the LLM, extracts associative
    anchors, traverses the memory graph, and injects relevant context.
    The LLM starts thinking with memories already present.

    Hybrid extraction: regex (always, 2ms) + optional local model (~250ms).

    All external dependencies are optional:
    - ``repository``: enables graph BFS + keyword search
    - ``vector_client``: enables semantic vector search
    - ``enable_model``: enables local MLX model extraction

    Without any backends, SYNAPSE is a no-op — safe to wire in before
    backends are ready.
    """

    def __init__(
        self,
        repository: Optional[MemoryRepository] = None,
        vector_client: Optional[VectorSearchClient] = None,
        config: Optional[BreatheConfig] = None,
        enable_model: bool = True,
    ):
        """
        Args:
            repository: Storage backend for graph BFS + keyword search.
                        If None, SYNAPSE skips graph traversal.
            vector_client: Semantic search client. If None, skips vector retrieval.
            config: BreatheConfig with language packs and tuning parameters.
                    Defaults to BreatheConfig() with EN + RU support.
            enable_model: Whether to enable local MLX model extraction (Phase 3).
        """
        self._repo = repository
        self._vector = vector_client
        self._config = config or BreatheConfig()
        self._enable_model = enable_model

        self._injector = ContextInjector(
            mode_budgets=self._config.mode_budgets,
            labels=self._config.labels,
        )
        self._extractor: Optional[AnchorExtractor] = None
        self._model_extractor = None
        self._known_concepts: dict[str, str] = {}
        self._initialized = False
        self._session_injected: set[str] = set()
        self.metrics = BreatheMetrics.get()

    async def initialize(self) -> None:
        """
        Load known concepts from storage and optionally prepare local model.
        Called once at session start — safe to call multiple times (idempotent).
        """
        if self._initialized:
            return

        try:
            if self._repo:
                self._known_concepts = await self._repo.get_concepts()
        except Exception as e:
            logger.warning(f"SYNAPSE: failed to load concepts from repo: {e}")
            self._known_concepts = {}

        # Build temporal / emotional patterns from all configured language packs
        temporal_patterns = [
            pack.temporal_pattern for pack in self._config.language_packs
        ]
        emotional_patterns = [
            pack.emotional_pattern for pack in self._config.language_packs
        ]

        self._extractor = AnchorExtractor(
            known_concepts=self._known_concepts,
            temporal_patterns=temporal_patterns,
            emotional_patterns=emotional_patterns,
        )

        if self._enable_model:
            try:
                from .model_extractor import ModelAnchorExtractor
                candidate = ModelAnchorExtractor()
                if candidate.available:
                    self._model_extractor = candidate
                    logger.info("SYNAPSE: model extractor available (lazy load on first use)")
            except Exception as e:
                logger.debug(f"SYNAPSE: model extractor unavailable: {e}")

        self._initialized = True
        self.metrics.known_concepts_count = len(self._known_concepts)
        self.metrics.model_available = self._model_extractor is not None
        logger.info(f"SYNAPSE initialized: {len(self._known_concepts)} known concepts")

    async def inject(self, messages: list) -> list:
        """
        Main entry point. Inject associative memory into messages.

        Inserts a ``<associative_memory>`` block into the last user message.
        If nothing relevant is found, returns messages unchanged.

        Args:
            messages: Conversation messages (system + user/assistant pairs).

        Returns:
            Messages with associative memory injected, or unchanged if nothing found.
        """
        if not self._repo and not self._vector:
            return messages

        if not self._initialized:
            await self.initialize()

        start_time = time.monotonic()

        try:
            user_message, tail = self._extract_context(messages)
            if not user_message:
                return messages

            regex_start = time.monotonic()
            result = self._extractor.extract(user_message)
            regex_ms = (time.monotonic() - regex_start) * 1000

            model_ms = 0.0
            model_triggered = False
            if self._model_extractor:
                from .model_extractor import should_use_model
                if should_use_model(result, self._config.model_trigger_threshold):
                    model_triggered = True
                    self.metrics.model_triggers += 1
                    model_start = time.monotonic()
                    model_anchors = self._model_extractor.extract(user_message)
                    model_ms = (time.monotonic() - model_start) * 1000
                    if model_anchors:
                        existing_texts = {a.text.lower() for a in result.anchors}
                        added = 0
                        for anchor in model_anchors:
                            if anchor.text.lower() not in existing_texts:
                                anchor = self._match_to_concepts(anchor)
                                result.anchors.append(anchor)
                                existing_texts.add(anchor.text.lower())
                                added += 1
                        logger.info(f"Model added {added} anchors → total {len(result.anchors)}")

            if not result.anchors:
                self.metrics.record_synapse_skip()
                return messages

            nodes = await self._traverse(result, user_message=user_message)
            if not nodes:
                return messages

            injection_text = self._injector.format_injection(
                nodes=nodes,
                mode=result.conversation_mode,
                anchors_text=[a.text for a in result.anchors],
            )
            if not injection_text:
                return messages

            messages = _insert_injection(messages, injection_text)

            injected_ids = []
            for node in nodes:
                if node.node_id:
                    self._session_injected.add(node.node_id)
                    injected_ids.append(node.node_id[:8])

            elapsed_ms = (time.monotonic() - start_time) * 1000

            self.metrics.record_synapse(SynapseEvent(
                timestamp=time.time(),
                anchors_count=len(result.anchors),
                matched_nodes=len(result.node_ids),
                injected_nodes=len(nodes),
                regex_ms=regex_ms,
                model_ms=model_ms,
                total_ms=elapsed_ms,
                mode=result.conversation_mode,
                model_triggered=model_triggered,
                user_message=user_message[:1000],
                anchors=[
                    AnchorDetail(
                        text=a.text, anchor_type=a.anchor_type,
                        confidence=a.confidence, source=a.source,
                        matched_node_id=a.matched_node_id,
                    ) for a in result.anchors
                ],
                nodes=[
                    NodeDetail(
                        concept=n.concept, node_type=n.node_type,
                        summary=(n.summary or "")[:500], importance=n.importance,
                        depth=n.depth, relation=n.relation,
                    ) for n in nodes
                ],
                injection_text=injection_text[:2000] if injection_text else "",
            ))
            self.metrics.record_anchors([a.text for a in result.anchors])

            logger.info(
                f"SYNAPSE: {len(result.anchors)} anchors → "
                f"{len(nodes)} nodes → injected ({elapsed_ms:.0f}ms)"
            )

        except Exception as e:
            logger.warning(f"SYNAPSE injection failed: {e}")
            # Never fail the request — just skip injection

        return messages

    def reset_session(self) -> None:
        """Clear session deduplication cache (call at session start)."""
        self._session_injected.clear()

    def _match_to_concepts(self, anchor: Anchor) -> Anchor:
        text_lower = anchor.text.lower().strip()
        for concept, node_id in self._known_concepts.items():
            concept_lower = concept.lower()
            if text_lower == concept_lower:
                anchor.matched_node_id = node_id
                anchor.source = "model+concept"
                anchor.confidence = 0.9
                break
            if text_lower in set(concept_lower.split()):
                anchor.matched_node_id = node_id
                anchor.source = "model+concept"
                anchor.confidence = 0.8
                break
        return anchor

    async def _traverse(
        self, anchor_result: AnchorResult, user_message: str = ""
    ) -> list[RetrievedNode]:
        nodes: list[RetrievedNode] = []

        # Strategy 1: Graph BFS from matched node IDs
        if self._repo and anchor_result.node_ids:
            try:
                graph_nodes = await self._repo.graph_bfs(anchor_result.node_ids)
                nodes.extend(graph_nodes)
            except Exception as e:
                logger.warning(f"SYNAPSE graph BFS failed: {e}")

        # Strategy 2: Vector search (semantic, highest quality)
        if self._vector and anchor_result.anchors:
            hub = self._config.hub_exclusions
            anchor_query = " ".join(
                a.text for a in anchor_result.anchors
                if a.text.lower() not in hub
            )
            if anchor_query.strip():
                try:
                    vector_nodes = await self._vector.search(anchor_query)
                    filtered = [n for n in vector_nodes if n.importance >= self._config.min_similarity]
                    nodes.extend(filtered)
                except Exception as e:
                    logger.warning(f"SYNAPSE vector search failed: {e}")

        # Strategy 3: Keyword search for unmatched anchors
        if self._repo:
            unmatched = [
                a for a in anchor_result.anchors
                if not a.matched_node_id and a.anchor_type in ("entity", "temporal", "theme")
            ]
            if unmatched:
                try:
                    kw_nodes = await self._repo.keyword_search([a.text for a in unmatched[:5]])
                    nodes.extend(kw_nodes)
                except Exception as e:
                    logger.warning(f"SYNAPSE keyword search failed: {e}")

        # Deduplicate
        seen: set[str] = set()
        unique: list[RetrievedNode] = []
        for node in nodes:
            if node.node_id not in seen:
                seen.add(node.node_id)
                unique.append(node)

        # Session dedup
        if self._session_injected:
            unique = [n for n in unique if n.node_id not in self._session_injected]

        # Hub exclusion
        hub = self._config.hub_exclusions
        unique = [n for n in unique if n.concept.lower() not in hub]

        # Relevance filter
        anchor_texts = {a.text.lower() for a in anchor_result.anchors}
        clean_anchors = {t for t in anchor_texts if t not in hub}
        if clean_anchors:
            anchor_words: set[str] = set()
            stopwords = self._config.stopwords
            for a in clean_anchors:
                for w in _WORD_RE.findall(a):
                    w_lower = w.lower()
                    if w_lower not in stopwords and len(w_lower) >= 3:
                        anchor_words.add(w_lower)

            if anchor_words:
                scored = []
                for node in unique:
                    score = _relevance_score(node, anchor_words, stopwords)
                    if score > 0:
                        scored.append((score, node))
                scored.sort(key=lambda x: (-x[0], -x[1].importance))
                unique = [node for _, node in scored]

        return unique[: self._config.max_injected_nodes]

    @staticmethod
    def _extract_context(messages: list) -> tuple[str, list[str]]:
        user_message = ""
        tail: list[str] = []
        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if not isinstance(content, str):
                continue
            if role == "user" and not user_message:
                user_message = content
            elif role in ("user", "assistant") and content and len(tail) < 3:
                tail.append(content[:500])
        return user_message, tail


def _relevance_score(
    node: RetrievedNode, anchor_words: set[str], stopwords: frozenset[str]
) -> float:
    """Score node relevance by keyword overlap with extracted anchors."""
    node_text = f"{node.concept} {node.summary or ''} {node.relation or ''}"
    node_words = set(w.lower() for w in _WORD_RE.findall(node_text)) - stopwords
    if not node_words or not anchor_words:
        return 0.0
    overlap = anchor_words & node_words
    if not overlap:
        anchor_stems = {w[:5] for w in anchor_words if len(w) >= 5}
        node_stems = {w[:5] for w in node_words if len(w) >= 5}
        stem_overlap = anchor_stems & node_stems
        if stem_overlap:
            overlap = stem_overlap
    return len(overlap) / max(len(anchor_words), 1)


def _insert_injection(messages: list, injection_text: str) -> list:
    """Prepend injection text to the last user message."""
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return messages

    new_messages = list(messages)
    original = new_messages[last_user_idx]
    content = original.get("content", "")

    if isinstance(content, list):
        injection_block = {"type": "text", "text": injection_text + "\n\n"}
        new_messages[last_user_idx] = {**original, "content": [injection_block] + content}
    else:
        new_messages[last_user_idx] = {**original, "content": injection_text + "\n\n" + content}

    return new_messages
