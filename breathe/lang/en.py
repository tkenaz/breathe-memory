"""English language pack for BREATHE."""
import re

STOPWORDS: frozenset[str] = frozenset({
    # Articles, prepositions, conjunctions
    "the", "and", "for", "that", "this", "with", "from", "have", "has",
    "was", "were", "been", "being", "are", "not", "but", "they", "them",
    "their", "what", "which", "when", "where", "how", "can", "will",
    "would", "should", "could", "just", "also", "more", "some", "than",
    "into", "about", "over", "after", "before", "between", "through",
    "during", "without", "again", "here", "there", "then", "now",
    "very", "really", "quite", "still", "already", "never", "always",
    "each", "every", "other", "another", "such", "only", "even",
    # Common verbs that match everything
    "work", "make", "use", "get", "set", "run", "add", "see", "say",
    "need", "want", "know", "think", "look", "come", "give", "take",
    # Generic nouns
    "time", "thing", "way", "day", "part", "point", "place", "case",
    "state", "fact", "line", "end", "start", "step", "type", "kind",
})

# Hub node names — too generic to be meaningful in retrieval
HUB_EXCLUSIONS: frozenset[str] = frozenset({
    "claude", "memory", "atlas",
})

# UI labels
LABELS: dict[str, str] = {
    "themes": "Themes",
    "insights": "Insights",
    "associative_memory_tag": "associative_memory",
}

# Temporal patterns (EN)
TEMPORAL_PATTERN = re.compile(
    r"\b(yesterday|today|tomorrow|last week|this week|last month|"
    r"recently|again|same as|remember when|like before|back then)\b",
    re.IGNORECASE,
)

# Emotional patterns (EN)
EMOTIONAL_PATTERN = re.compile(
    r"\b(tired|hurts|headache|frustrated|angry|annoyed|"
    r"happy|amazing|awesome|fuck|shit|damn|"
    r"sad|miss|worried|anxious|"
    r"love|hug|gentle|warm)\b",
    re.IGNORECASE,
)
