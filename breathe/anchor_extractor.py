"""
Anchor Extractor — Phase 1 (Regex MVP).

Extracts associative anchors from user message + conversation tail.
Anchors are entry points for graph traversal in SYNAPSE.

Phase 1: Regex + known concepts from MemoryRepository.
Phase 2 (optional): Local MLX model via ModelAnchorExtractor.

Yes, regex is dumb. It's a skeleton — the pipeline matters more than the extractor.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Anchor:
    """A single associative anchor extracted from text."""

    text: str
    anchor_type: str  # 'entity', 'temporal', 'theme', 'technical', 'emotional'
    confidence: float = 0.5  # 0.0–1.0
    source: str = "regex"  # 'regex', 'known_concept', 'model'
    matched_node_id: Optional[str] = None  # UUID if matched to memory_nodes


@dataclass
class AnchorResult:
    """Result of anchor extraction."""

    anchors: list[Anchor] = field(default_factory=list)
    raw_text: str = ""
    conversation_mode: str = "balanced"  # casual / work / deep / balanced

    @property
    def entity_anchors(self) -> list[Anchor]:
        return [a for a in self.anchors if a.anchor_type == "entity"]

    @property
    def has_temporal(self) -> bool:
        return any(a.anchor_type == "temporal" for a in self.anchors)

    @property
    def node_ids(self) -> list[str]:
        """UUIDs of matched memory_nodes."""
        return [a.matched_node_id for a in self.anchors if a.matched_node_id]


# --- Shared technical pattern (language-agnostic) ---
TECH_PATTERN = re.compile(
    r"(?:"
    r"[a-zA-Z_]\w+\.(?:py|ts|tsx|js|yaml|json|md|sql|sh)|"  # file paths
    r"(?:def |class |function |const |import |from )\w+|"  # code keywords
    r"(?:localhost:\d+|https?://\S+)|"  # URLs
    r"(?:MCP|API|SSE|JWT|OAuth|CORS|FastAPI|React|PostgreSQL|Redis)\b"  # tech terms
    r")",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})\b",
    re.IGNORECASE,
)


class AnchorExtractor:
    """
    Extract associative anchors from user message.

    Phase 1: Regex + known concepts dictionary.
    The extractor loads known concepts from MemoryRepository at init
    and matches them against incoming messages.

    Supports multiple language packs — temporal and emotional patterns
    are loaded from the configured BreatheConfig.
    """

    def __init__(
        self,
        known_concepts: Optional[dict[str, str]] = None,
        temporal_patterns: Optional[list[re.Pattern]] = None,
        emotional_patterns: Optional[list[re.Pattern]] = None,
    ):
        """
        Args:
            known_concepts: ``{concept_text: node_uuid}`` dict from MemoryRepository.
                           Loaded once at session start via Synapse.initialize().
            temporal_patterns: Compiled regex patterns for temporal references.
                               Defaults to English patterns.
            emotional_patterns: Compiled regex patterns for emotional language.
                                Defaults to English patterns.
        """
        self.known_concepts = known_concepts or {}

        # Build fast regex from known concepts (sorted by length for greedy match)
        if self.known_concepts:
            escaped = [
                re.escape(c)
                for c in sorted(self.known_concepts.keys(), key=len, reverse=True)
            ]
            self._concept_pattern: Optional[re.Pattern] = re.compile(
                r"\b(" + "|".join(escaped) + r")\b",
                re.IGNORECASE,
            )
        else:
            self._concept_pattern = None

        # Default to EN patterns if none supplied
        if temporal_patterns is None:
            from .lang.en import TEMPORAL_PATTERN as EN_T
            temporal_patterns = [EN_T]

        if emotional_patterns is None:
            from .lang.en import EMOTIONAL_PATTERN as EN_E
            emotional_patterns = [EN_E]

        self._temporal_patterns = temporal_patterns
        self._emotional_patterns = emotional_patterns

    def extract(
        self,
        message: str,
        conversation_tail: Optional[list[str]] = None,
    ) -> AnchorResult:
        """
        Extract anchors from user message + optional conversation tail.

        Args:
            message: Current user message text.
            conversation_tail: Last 2–3 messages for context (strings).

        Returns:
            AnchorResult with extracted anchors and detected conversation mode.
        """
        result = AnchorResult(raw_text=message)

        # Combine message with tail for broader concept matching
        full_text = message
        if conversation_tail:
            full_text = message + " " + " ".join(conversation_tail[-3:])

        # 1. Match known concepts (highest confidence — direct graph entry points)
        if self._concept_pattern:
            for match in self._concept_pattern.finditer(full_text):
                concept_text = match.group(0).lower()
                node_id = next(
                    (uid for key, uid in self.known_concepts.items()
                     if key.lower() == concept_text),
                    None,
                )
                result.anchors.append(Anchor(
                    text=match.group(0),
                    anchor_type="entity",
                    confidence=0.9,
                    source="known_concept",
                    matched_node_id=node_id,
                ))

        # 2. Temporal anchors
        for pattern in (*self._temporal_patterns, DATE_PATTERN):
            for match in pattern.finditer(message):
                result.anchors.append(Anchor(
                    text=match.group(0),
                    anchor_type="temporal",
                    confidence=0.7,
                ))

        # 3. Technical anchors (message only — tail brings too much noise)
        for match in TECH_PATTERN.finditer(message):
            text = match.group(0).strip()
            if len(text) > 2:
                result.anchors.append(Anchor(
                    text=text,
                    anchor_type="technical",
                    confidence=0.5,
                ))

        # 4. Emotional anchors
        emotional_count = 0
        for pattern in self._emotional_patterns:
            for match in pattern.finditer(message):
                emotional_count += 1
                result.anchors.append(Anchor(
                    text=match.group(0),
                    anchor_type="emotional",
                    confidence=0.6,
                ))

        # Exclamation density as emotional signal
        if message.count("!") / max(len(message), 1) > 0.02:
            emotional_count += 1

        result.conversation_mode = self._detect_mode(message, result.anchors, emotional_count)
        result.anchors = self._deduplicate(result.anchors)

        logger.info(
            f"Anchors: {len(result.anchors)} extracted "
            f"({len(result.node_ids)} matched nodes), "
            f"mode={result.conversation_mode}"
        )

        return result

    def _detect_mode(self, message: str, anchors: list[Anchor], emotional_count: int) -> str:
        tech_count = sum(1 for a in anchors if a.anchor_type == "technical")
        if tech_count >= 3:
            return "work"
        if emotional_count >= 2:
            return "deep"
        if len(message) < 100 and tech_count == 0:
            return "casual"
        return "balanced"

    @staticmethod
    def _deduplicate(anchors: list[Anchor]) -> list[Anchor]:
        """Keep highest-confidence anchor for each unique text (case-insensitive)."""
        seen: dict[str, Anchor] = {}
        for anchor in anchors:
            key = anchor.text.lower()
            if key not in seen or anchor.confidence > seen[key].confidence:
                seen[key] = anchor
        return list(seen.values())
