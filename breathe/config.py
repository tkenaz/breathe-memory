"""
BreatheConfig — unified configuration for BREATHE.

All tuneable parameters live here. Language packs plug in as dicts,
making BREATHE easy to extend to any language without changing core logic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .lang.en import (
    STOPWORDS as EN_STOPWORDS,
    HUB_EXCLUSIONS as EN_HUB_EXCLUSIONS,
    LABELS as EN_LABELS,
    TEMPORAL_PATTERN as EN_TEMPORAL,
    EMOTIONAL_PATTERN as EN_EMOTIONAL,
)


@dataclass
class LanguagePack:
    """
    Language configuration for anchor extraction and context injection.

    To add a new language, create a LanguagePack and pass it to BreatheConfig.

    Example::

        from breathe.config import LanguagePack, BreatheConfig
        import re

        my_pack = LanguagePack(
            code="de",
            stopwords={"der", "die", "das", "und", "ist", ...},
            hub_exclusions={"claude", "speicher"},
            temporal_pattern=re.compile(r"\\b(gestern|heute|morgen)\\b", re.I),
            emotional_pattern=re.compile(r"\\b(traurig|glücklich|wütend)\\b", re.I),
            labels={"themes": "Themen", "insights": "Erkenntnisse"},
        )
        config = BreatheConfig(language_packs=[my_pack], default_language="de")
    """

    code: str
    stopwords: frozenset[str]
    hub_exclusions: frozenset[str]
    temporal_pattern: re.Pattern
    emotional_pattern: re.Pattern
    labels: dict[str, str]


# Pre-built packs
ENGLISH = LanguagePack(
    code="en",
    stopwords=EN_STOPWORDS,
    hub_exclusions=EN_HUB_EXCLUSIONS,
    temporal_pattern=EN_TEMPORAL,
    emotional_pattern=EN_EMOTIONAL,
    labels=EN_LABELS,
)

@dataclass
class BreatheConfig:
    """
    Master configuration for BREATHE.

    All settings have sensible defaults. The most common customizations are:
    - ``language_packs``: which languages to support (default: EN)
    - ``default_language``: primary language for UI strings
    - ``min_similarity``: threshold for vector search results (0–1)
    - ``max_injected_nodes``: upper limit on nodes per injection (default 15)
    - ``enable_model_extractor``: whether to use local MLX model (default True)

    Token budgets control how much memory is injected per conversation mode.
    Adjust them based on your model's context window and use case.
    """

    # --- Language ---
    language_packs: list[LanguagePack] = field(
        default_factory=lambda: [ENGLISH]
    )
    default_language: str = "en"

    # --- SYNAPSE ---
    min_similarity: float = 0.55
    """Minimum vector similarity score to accept (below = noise)."""

    max_injected_nodes: int = 15
    """Maximum nodes per injection pass."""

    enable_model_extractor: bool = True
    """Whether to use local MLX model for enhanced anchor extraction (Phase 3)."""

    model_trigger_threshold: int = 5
    """
    Model extractor fires when regex finds fewer matched nodes than this.
    Lower = model runs more often (slower but richer extraction).
    """

    # --- Token budgets by conversation mode ---
    mode_budgets: dict[str, int] = field(
        default_factory=lambda: {
            "casual": 1500,
            "work": 2500,
            "deep": 4000,
            "balanced": 2000,
        }
    )

    # --- GraphCompactor ---
    compactor_model: str = "claude-sonnet-4-6"
    compactor_fallback_model: str = "claude-haiku-4-5-20251001"
    min_tokens_to_compress: int = 300
    protected_messages_normal: int = 10
    protected_messages_with_code: int = 5

    # --- Metrics ---
    metrics_history_size: int = 200
    """How many events to keep in the rolling metrics window."""

    # --- Computed (built from language_packs) ---

    @property
    def stopwords(self) -> frozenset[str]:
        """Union of stopwords from all configured language packs."""
        combined: set[str] = set()
        for pack in self.language_packs:
            combined |= pack.stopwords
        return frozenset(combined)

    @property
    def hub_exclusions(self) -> frozenset[str]:
        """Union of hub exclusions from all configured language packs."""
        combined: set[str] = set()
        for pack in self.language_packs:
            combined |= pack.hub_exclusions
        return frozenset(combined)

    @property
    def labels(self) -> dict[str, str]:
        """UI labels from the default language pack."""
        for pack in self.language_packs:
            if pack.code == self.default_language:
                return pack.labels
        return self.language_packs[0].labels if self.language_packs else {}

    def get_pack(self, code: str) -> Optional[LanguagePack]:
        """Return language pack by code, or None if not configured."""
        for pack in self.language_packs:
            if pack.code == code:
                return pack
        return None
