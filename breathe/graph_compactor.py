"""
GraphCompactor — structured graph extraction for context compression.

The exhale of BREATHE. Fires when the context window approaches its limit.
Instead of a lossy narrative summary, extracts a structured graph of topics,
decisions, open questions, and artifacts. The LLM decides what matters.

Design principle: LLM memory is LLM's decision, not a summarizer's interpretation.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Optional

from .interfaces import LLMClient
from .session_graph import SessionGraph
from .metrics import BreatheMetrics, CompactionEvent

logger = logging.getLogger(__name__)

TOOL_USE_PATTERN = re.compile(r"\[tool_use\]\s*(\S+)\[/tool_use\]")
TOOL_RESULT_PATTERN = re.compile(r"\[tool_result\](.*?)\[/tool_result\]", re.DOTALL)

EXTRACTION_PROMPT = """You are extracting a structured context graph from a conversation.

This replaces linear compression. You decide what matters. Be selective, not exhaustive.

From the messages below, extract:

## Topics [weight 0.0-1.0]
Active topics of conversation. Weight = how central they are right now.
Format: - [topic_name] [weight] description | connected_topics

## Decisions
What was decided or concluded. One line each.
Format: - decision text

## Open
Unresolved questions or pending items.
Format: - question or pending item

## Artifacts
Files, code, configs created or modified.
Format: - name: what it is

## Context
Emotional state, situational context that affects interpretation.
Format: - context note

## Dropped
What you chose NOT to preserve, and why (one line). This makes the choice conscious.
Format: - what was dropped (why)

IMPORTANT:
- Tool calls: drop the raw JSON. Keep only "searched X → found Y" or "modified file Z".
- Keep the language of the original conversation.
- Be concise. This must fit in ~10k tokens.
- Weight topics by current relevance, not chronological order.

Messages to extract from:
"""


class GraphCompactor:
    """
    Drop-in replacement for any narrative summarizer.

    Compresses older conversation history into a structured graph instead
    of a lossy prose summary. Preserves the semantic structure of what happened.

    Usage::

        compactor = GraphCompactor(llm_client=my_llm_client)
        result = await compactor.compress(messages)
        if result["compressed"]:
            messages = result["compressed_messages"]
    """

    def __init__(
        self,
        llm_client: LLMClient,
        min_tokens_to_compress: int = 300,
        protected_messages_normal: int = 10,
        protected_messages_with_code: int = 5,
    ):
        """
        Args:
            llm_client: LLM client for graph extraction. Use AnthropicLLMClient
                        or implement LLMClient for any other provider.
            min_tokens_to_compress: Skip compression if older messages have fewer
                                    tokens than this threshold.
            protected_messages_normal: How many recent assistant turns to keep
                                       intact (not compressed).
            protected_messages_with_code: Protected turns when code/tools detected.
        """
        self._llm = llm_client
        self.MIN_TOKENS_TO_COMPRESS = min_tokens_to_compress
        self.PROTECTED_MESSAGES_NORMAL = protected_messages_normal
        self.PROTECTED_MESSAGES_WITH_CODE = protected_messages_with_code

    @staticmethod
    def _count_tokens_rough(text: str) -> int:
        return len(text) // 4

    @staticmethod
    def _has_tool_calls(messages: list) -> bool:
        recent = messages[-10:] if len(messages) > 10 else messages
        return any(
            "```" in msg.get("content", "")
            or "[tool_use]" in msg.get("content", "")
            or "def " in msg.get("content", "")
            for msg in recent
        )

    def _split_messages(self, messages: list) -> tuple[list, list]:
        """Split into compressible (older) and protected (recent) zones."""
        has_code = self._has_tool_calls(messages)
        protected_count = (
            self.PROTECTED_MESSAGES_WITH_CODE if has_code else self.PROTECTED_MESSAGES_NORMAL
        )
        assistant_indices = [
            i for i, msg in enumerate(messages) if msg.get("role") == "assistant"
        ]
        if len(assistant_indices) <= protected_count:
            return [], messages
        protected_start_idx = assistant_indices[-protected_count]
        return messages[:protected_start_idx], messages[protected_start_idx:]

    @staticmethod
    def _preprocess_tool_calls(messages: list) -> list:
        """Compress raw tool call JSON before sending to extraction."""
        processed = []
        for msg in messages:
            content = msg.get("content", "")
            content = TOOL_USE_PATTERN.sub(r"[tool: \1]", content)

            def compress_result(match: re.Match) -> str:
                result_text = match.group(1).strip()
                if len(result_text) > 200:
                    return f"[result: {result_text[:200]}...]"
                return f"[result: {result_text}]"

            content = TOOL_RESULT_PATTERN.sub(compress_result, content)
            if content.strip():
                processed.append({**msg, "content": content})
        return processed

    @staticmethod
    def _format_for_extraction(messages: list) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 3000:
                content = content[:3000] + "... [truncated]"
            lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)

    async def compress(
        self, messages: list, conversation_style: str = "balanced"
    ) -> dict:
        """
        Extract structured graph from compressible messages.

        Args:
            messages: Full conversation messages list.
            conversation_style: Conversation style hint (unused currently,
                                reserved for future extraction tuning).

        Returns:
            Dict with keys:
            - ``compressed``: bool
            - ``compressed_messages``: list (use this to replace ``messages``)
            - ``messages``: same as compressed_messages (alias)
            - ``metadata``: compression stats
            - ``session_graph``: SessionGraph instance (if compressed)
        """
        compressible, protected = self._split_messages(messages)

        if not compressible or len(compressible) < 3:
            return self._no_op(messages, reason="too_few_messages")

        original_tokens = sum(
            self._count_tokens_rough(msg.get("content", ""))
            for msg in compressible
        )

        if original_tokens < self.MIN_TOKENS_TO_COMPRESS:
            return self._no_op(messages, reason="below_threshold")

        compact_start = time.monotonic()
        logger.info(
            f"Graph compaction: {len(compressible)} messages, "
            f"~{original_tokens} tokens → extracting..."
        )

        processed = self._preprocess_tool_calls(compressible)
        formatted = self._format_for_extraction(processed)
        prompt = EXTRACTION_PROMPT + formatted

        extracted_text = await self._llm.complete(prompt)
        if not extracted_text:
            logger.error("Graph extraction failed, returning original messages")
            return self._no_op(messages, reason="extraction_failed")

        session_graph = SessionGraph.from_structured_text(extracted_text)
        logger.info(f"Extracted graph: {session_graph}")

        compressed_tokens = self._count_tokens_rough(extracted_text)
        saved_tokens = original_tokens - compressed_tokens
        compression_ratio = (
            1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0
        )

        compressed_message = {
            "role": "system",
            "content": (
                f"[SESSION GRAPH — extracted from {len(compressible)} messages]\n\n"
                f"{extracted_text}"
            ),
        }

        final_messages = [compressed_message] + protected
        compact_ms = (time.monotonic() - compact_start) * 1000

        BreatheMetrics.get().record_compaction(CompactionEvent(
            timestamp=time.time(),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            saved_tokens=saved_tokens,
            compression_ratio=compression_ratio,
            graph_nodes=session_graph.node_count,
            graph_edges=session_graph.edge_count,
            strategy=f"graph_{conversation_style}",
            fallback_used=False,
            duration_ms=compact_ms,
            extracted_text=extracted_text[:3000] if extracted_text else "",
        ))

        logger.info(
            f"Graph compaction complete: {original_tokens} → {compressed_tokens} tokens "
            f"(saved {saved_tokens}, {compression_ratio:.0%}), "
            f"{session_graph.node_count} nodes, {session_graph.edge_count} edges"
        )

        return {
            "compressed": True,
            "messages": final_messages,
            "compressed_messages": final_messages,
            "session_graph": session_graph,
            "metadata": {
                "original_messages": len(compressible),
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "saved_tokens": saved_tokens,
                "compression_ratio": round(compression_ratio, 2),
                "protected_messages": len(protected),
                "strategy": f"graph_{conversation_style}",
                "graph_nodes": session_graph.node_count,
                "graph_edges": session_graph.edge_count,
                "compressed_at": datetime.utcnow().isoformat(),
            },
        }

    @staticmethod
    def _no_op(messages: list, reason: str) -> dict:
        return {
            "compressed": False,
            "messages": messages,
            "compressed_messages": messages,
            "metadata": {"reason": reason},
        }
