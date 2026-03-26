"""Language packs for BREATHE anchor extraction and injection."""
from .en import (
    STOPWORDS as EN_STOPWORDS,
    HUB_EXCLUSIONS as EN_HUB_EXCLUSIONS,
    LABELS as EN_LABELS,
    TEMPORAL_PATTERN as EN_TEMPORAL,
    EMOTIONAL_PATTERN as EN_EMOTIONAL,
)

__all__ = [
    "EN_STOPWORDS", "EN_HUB_EXCLUSIONS", "EN_LABELS", "EN_TEMPORAL", "EN_EMOTIONAL",
]
