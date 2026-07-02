"""
Lightweight lexical retrieval over the SHL Individual Test Solutions catalog.

Why BM25 and not embeddings/a vector DB: the catalog is ~380 rows of short,
keyword-dense text (job titles, tool names, skill names — "Java", "SQL",
"Accounts Payable"). BM25 handles exact/near-exact keyword matches on this
kind of vocabulary at least as well as embeddings, needs no external API
calls or vector infra, and is fully deterministic/debuggable, which matters
for a service that's graded on hallucination rate and Recall@10. If the
catalog grows to a scale where lexical recall starts missing paraphrased
queries, swap this module for an embedding index without touching callers
(`search()` is the only public surface other modules depend on).
"""
import json
import re
from pathlib import Path
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class CatalogEntry:
    name: str
    url: str
    description: str
    job_levels: list[str]
    languages: list[str]
    duration_minutes: int | None
    test_type: list[str]
    test_type_labels: list[str]
    remote_testing: bool | None
    adaptive_irt: bool | None

    def to_recommendation(self) -> dict:
        return {"name": self.name, "url": self.url, "test_type": self.test_type}

    def searchable_text(self) -> str:
        parts = [
            self.name,
            self.description or "",
            " ".join(self.job_levels),
            " ".join(self.languages),
            " ".join(self.test_type_labels),
        ]
        return " ".join(p for p in parts if p)


class CatalogIndex:
    def __init__(self, catalog_path: str | Path):
        self.entries: list[CatalogEntry] = self._load(catalog_path)
        if not self.entries:
            raise ValueError(f"Loaded zero catalog entries from {catalog_path}")
        self._corpus_tokens = [_tokenize(e.searchable_text()) for e in self.entries]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._url_index = {e.url: e for e in self.entries}

    @staticmethod
    def _load(path: str | Path) -> list[CatalogEntry]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        entries = []
        for r in raw:
            entries.append(CatalogEntry(
                name=r["name"],
                url=r["url"],
                description=r.get("description") or "",
                job_levels=r.get("job_levels") or [],
                languages=r.get("languages") or [],
                duration_minutes=r.get("duration_minutes"),
                test_type=r.get("test_type") or [],
                test_type_labels=r.get("test_type_labels") or [],
                remote_testing=r.get("remote_testing"),
                adaptive_irt=r.get("adaptive_irt"),
            ))
        return entries

    def search(self, query: str, top_k: int = 10) -> list[CatalogEntry]:
        """Return the top_k catalog entries most relevant to `query`."""
        if not query.strip():
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked_idx[: top_k * 3]:  # over-fetch, then drop zero-score noise
            if scores[i] <= 0:
                break
            results.append(self.entries[i])
        return results[:top_k]

    def get_by_url(self, url: str) -> CatalogEntry | None:
        return self._url_index.get(url)

    def filter(self, entries: list[CatalogEntry], *,
               max_duration_minutes: int | None = None,
               test_types: list[str] | None = None,
               remote_only: bool = False) -> list[CatalogEntry]:
        """Apply hard constraints on top of a relevance-ranked candidate list."""
        out = entries
        if max_duration_minutes is not None:
            out = [e for e in out
                   if e.duration_minutes is None or e.duration_minutes <= max_duration_minutes]
        if test_types:
            wanted = set(t.upper() for t in test_types)
            out = [e for e in out if wanted.intersection(e.test_type)]
        if remote_only:
            out = [e for e in out if e.remote_testing is True]
        return out
