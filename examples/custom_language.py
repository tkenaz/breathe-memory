"""
Custom language pack — adding German support to BREATHE.

Demonstrates how to extend BREATHE to any language in ~20 lines.
"""
import asyncio
import re
from breathe import Synapse, BreatheConfig, LanguagePack, ENGLISH, RetrievedNode
from breathe.interfaces import MemoryRepository


# 1. Define a language pack
GERMAN = LanguagePack(
    code="de",
    stopwords=frozenset({
        "der", "die", "das", "und", "ist", "in", "zu", "von", "mit",
        "für", "auf", "an", "aus", "bei", "nach", "über", "unter",
        "nicht", "auch", "noch", "aber", "oder", "wenn", "dann",
        "ich", "du", "er", "sie", "wir", "ihr", "sie", "es",
        "sein", "haben", "werden", "können", "müssen", "sollen",
    }),
    hub_exclusions=frozenset({"claude", "speicher", "system"}),
    temporal_pattern=re.compile(
        r"\b(gestern|heute|morgen|letzte Woche|diese Woche|neulich|wieder|damals)\b",
        re.IGNORECASE,
    ),
    emotional_pattern=re.compile(
        r"\b(müde|Schmerzen|frustriert|wütend|glücklich|toll|scheiße|traurig|vermisse|liebe)\b",
        re.IGNORECASE,
    ),
    labels={
        "themes": "Themen",
        "insights": "Erkenntnisse",
        "associative_memory_tag": "assoziatives_gedächtnis",
    },
)


# 2. Minimal stub repo for demo
class MinimalRepo(MemoryRepository):
    async def get_concepts(self):
        return {"FastAPI": "uuid-1", "Python": "uuid-2"}

    async def graph_bfs(self, start_ids, **kwargs):
        return []

    async def keyword_search(self, keywords, limit=5):
        return [
            RetrievedNode(
                node_id=f"kw-{i}", concept=kw, node_type="memory_keyword",
                summary=f"Frühere Diskussion über {kw}", importance=0.6, depth=0,
                memory_content=f"Wir haben {kw} letzte Woche ausführlich besprochen.",
            )
            for i, kw in enumerate(keywords[:2])
        ]


async def main():
    # 3. Use in BreatheConfig — works with EN + DE simultaneously
    config = BreatheConfig(
        language_packs=[ENGLISH, GERMAN],
        default_language="de",
    )

    synapse = Synapse(repository=MinimalRepo(), config=config, enable_model=False)
    await synapse.initialize()

    messages = [
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Wie soll ich meine FastAPI Endpunkte strukturieren?"},
    ]

    enriched = await synapse.inject(messages)
    print("Enriched message (German):")
    print(enriched[-1]["content"])


if __name__ == "__main__":
    asyncio.run(main())
