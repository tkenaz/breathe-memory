"""
Context Injector — formats retrieved subgraph into injection text.

Takes nodes from SYNAPSE traversal and produces a structured text block
for context window injection.

The format uses <associative_memory> tags so the LLM perceives this
as remembered context, not tool output — "I remember" not "I was told."
"""
from __future__ import annotations

import logging
from typing import Optional

from .interfaces import RetrievedNode

logger = logging.getLogger(__name__)

# Default token budgets per conversation mode.
# Override via BreatheConfig.mode_budgets.
DEFAULT_MODE_BUDGETS: dict[str, int] = {
    "casual": 1500,
    "work": 2500,
    "deep": 4000,
    "balanced": 2000,
}


class ContextInjector:
    """
    Formats retrieved nodes into an injection text block.

    Respects per-mode token budgets. Output is structured but natural —
    the LLM should feel this as memory, not structured data.

    Labels (e.g. "Themes", "Insights") come from the active language pack.
    Pass a ``labels`` dict to override defaults.
    """

    def __init__(
        self,
        mode_budgets: Optional[dict[str, int]] = None,
        labels: Optional[dict[str, str]] = None,
        memory_tag: str = "associative_memory",
    ):
        """
        Args:
            mode_budgets: Token budget per conversation mode.
                          Defaults to DEFAULT_MODE_BUDGETS.
            labels: UI strings for section headers.
                    Keys: ``"themes"``, ``"insights"``.
                    Defaults to English.
            memory_tag: XML tag wrapping the injection block.
        """
        self._budgets = mode_budgets or DEFAULT_MODE_BUDGETS
        self._labels = labels or {"themes": "Themes", "insights": "Insights"}
        self._memory_tag = memory_tag

    def format_injection(
        self,
        nodes: list[RetrievedNode],
        mode: str = "balanced",
        anchors_text: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Format nodes into injection text.

        Args:
            nodes: Retrieved nodes from SYNAPSE traversal.
            mode: Conversation mode for budget selection.
            anchors_text: Original anchor texts (unused currently, reserved for
                          future "Connection to now" section).

        Returns:
            Formatted injection string, or None if nothing to inject.
        """
        if not nodes:
            return None

        budget_tokens = self._budgets.get(mode, 2000)
        budget_chars = budget_tokens * 4  # rough char-to-token ratio

        sections: list[str] = []

        # Group nodes by type
        entities = [n for n in nodes if n.node_type == "entity"]
        themes = [n for n in nodes if n.node_type == "theme"]
        insights = [n for n in nodes if n.node_type == "insight"]
        events = [n for n in nodes if n.node_type == "event"]

        # Events and entities carry the richest context
        for node in events[:3]:
            section = self._format_event(node)
            if section:
                sections.append(section)

        if entities:
            lines = []
            for node in entities[:5]:
                line = f"**{node.concept}**"
                if node.summary:
                    line += f" — {node.summary}"
                if node.relation:
                    line += f" ({node.relation})"
                lines.append(line)
            sections.append("\n".join(lines))

        if themes:
            theme_label = self._labels.get("themes", "Themes")
            theme_lines = [f"{theme_label}: {', '.join(n.concept for n in themes[:4])}"]
            for node in themes[:2]:
                if node.summary:
                    theme_lines.append(f"- {node.concept}: {node.summary}")
            sections.append("\n".join(theme_lines))

        if insights:
            insight_label = self._labels.get("insights", "Insights")
            insight_lines = [
                f"- {node.summary or node.concept}" for node in insights[:3]
            ]
            sections.append(f"{insight_label}:\n" + "\n".join(insight_lines))

        # Raw memory content (from vector/keyword search)
        memories_with_content = [n for n in nodes if n.memory_content]
        for node in memories_with_content[:3]:
            content = node.memory_content
            if len(content) > 1500:
                content = content[:1500] + "..."
            sections.append(content)

        if not sections:
            return None

        body = "\n\n".join(sections)

        if len(body) > budget_chars:
            body = body[:budget_chars] + "\n[...]"

        injection = f"<{self._memory_tag}>\n{body}\n</{self._memory_tag}>"

        logger.info(
            f"Injection: {len(nodes)} nodes → {len(injection)} chars "
            f"(budget: {budget_chars} chars, mode: {mode})"
        )

        return injection

    @staticmethod
    def _format_event(node: RetrievedNode) -> Optional[str]:
        parts = [f"### {node.concept}"]
        if node.summary:
            parts.append(node.summary)
        if node.memory_content:
            content = node.memory_content
            if len(content) > 1500:
                content = content[:1500] + "..."
            parts.append(content)
        return "\n".join(parts) if len(parts) > 1 else None
