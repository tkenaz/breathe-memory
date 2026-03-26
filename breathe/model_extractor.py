"""
Model-based Anchor Extractor — Phase 3 of BREATHE.

Uses a local small language model (Qwen3-1.7B abliterated, MLX 4-bit) for
contextual anchor extraction. Runs on Apple Silicon via MLX framework.

Designed as an enhancement layer ON TOP of the regex extractor:
- Regex is always fast (2ms) and catches known concepts
- Model adds contextual understanding (~250ms) when regex is insufficient
- Hybrid: regex always runs, model fires when regex finds <N matched nodes

The model is loaded lazily on first call and kept in memory (~1.2GB).
MLX dependency is optional — if not installed, this module is a no-op.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from .anchor_extractor import Anchor, AnchorResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "mlx-community/Josiefied-Qwen3-1.7B-abliterated-v1-4bit"

EXTRACTION_PROMPT = """You help another AI model retrieve relevant information from memory. When you see a user message, you need to identify which words are maximally informative — the key nodes of the phrase. The message may be in any language.

CRITICAL: Every keyword you return MUST actually appear in the message. Do NOT invent or hallucinate words that are not present. Only extract what is written.

Extract:
- entities: proper nouns — company names, product names, people, cities, projects, tools
- themes: abstract topics being discussed
- emotional: emotional state words ONLY if very strong (skip mild emotions)

Return ONLY a JSON object: {{"entities": [...], "themes": [...], "emotional": [...]}}
No explanation, no markdown, no thinking.
Message: {message}
JSON:"""


class ModelAnchorExtractor:
    """
    Local model anchor extraction via MLX.

    Lazy-loads model on first call. Kept in memory for subsequent calls.
    Install extras to enable: ``pip install breathe-memory[mlx]``
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Check if MLX is installed without loading the model."""
        if self._available is None:
            try:
                import mlx_lm  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.debug("mlx_lm not installed — model extractor disabled")
        return self._available

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            return False
        try:
            from mlx_lm import load
            start = time.monotonic()
            self._model, self._tokenizer = load(self.model_id)
            elapsed = (time.monotonic() - start) * 1000
            logger.info(f"Model extractor loaded: {self.model_id} ({elapsed:.0f}ms)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model extractor: {e}")
            self._available = False
            return False

    def extract(self, message: str, max_tokens: int = 100) -> list[Anchor]:
        """
        Extract anchors from message using the local model.

        Args:
            message: User message text.
            max_tokens: Max generation tokens (keep low for speed).

        Returns:
            Validated list of Anchor objects.
        """
        if not self._ensure_loaded():
            return []
        try:
            from mlx_lm import generate
            prompt = EXTRACTION_PROMPT.format(message=message[:500])
            start = time.monotonic()
            raw = generate(
                self._model, self._tokenizer,
                prompt=prompt, max_tokens=max_tokens, verbose=False,
            )
            elapsed = (time.monotonic() - start) * 1000
            anchors = _parse_response(raw)
            anchors = _validate_against_message(anchors, message)
            logger.info(f"Model extraction: {len(anchors)} anchors in {elapsed:.0f}ms")
            return anchors
        except Exception as e:
            logger.warning(f"Model extraction failed: {e}")
            return []


def should_use_model(
    regex_result: AnchorResult,
    threshold: int = 5,
) -> bool:
    """
    Decide whether to invoke the model based on regex results.

    Returns True when regex found fewer matched nodes than ``threshold``
    (meaning the message likely contains concepts the regex doesn't know about).
    Also skips very short messages (greetings, "ok", etc.).
    """
    if len(regex_result.raw_text) < 15:
        return False
    return len(regex_result.node_ids) < threshold


def _parse_response(raw: str) -> list[Anchor]:
    anchors: list[Anchor] = []
    raw = raw.strip()
    start_idx = raw.find("{")
    if start_idx == -1:
        return anchors

    depth = 0
    end_idx = start_idx
    for i in range(start_idx, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    try:
        data = json.loads(raw[start_idx:end_idx])
    except json.JSONDecodeError:
        return anchors

    type_map = {"entities": "entity", "themes": "theme", "emotional": "emotional"}
    for key, anchor_type in type_map.items():
        items = data.get(key, [])
        if isinstance(items, str) and items:
            items = [items]
        if not isinstance(items, list):
            continue
        for item in items:
            text = ""
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = (item.get("value") or item.get("name") or item.get("text") or "").strip()
            if text and len(text) > 1:
                anchors.append(Anchor(
                    text=text, anchor_type=anchor_type, confidence=0.7, source="model",
                ))
    return anchors


def _validate_against_message(anchors: list[Anchor], message: str) -> list[Anchor]:
    """Drop anchors not actually present in the message (hallucination guard).

    Uses stem matching (first 4 chars) to handle morphological variants.
    """
    msg_lower = message.lower()
    validated = []
    for a in anchors:
        words = a.text.lower().split()
        found = True
        for w in words:
            if w in msg_lower:
                continue
            stem = w[:4] if len(w) >= 4 else w
            if stem in msg_lower:
                continue
            found = False
            break
        if found:
            validated.append(a)
        else:
            logger.debug(f"Hallucination dropped: '{a.text}' (not in message)")
    return validated
