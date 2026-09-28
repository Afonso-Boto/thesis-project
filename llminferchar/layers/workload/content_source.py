"""
Content source for the workload generator
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CORPUS_PATH = (
    _REPO_ROOT / "llminferchar"/  "data" / "corpus" / "wikitext103_prose_v1.jsonl"
)


class CorpusError(Exception):
    pass


# Corpus pool with pinned JSONL corpus with hash verification
@dataclass
class CorpusPool:
    """A paragraph corpus loaded from a pinned JSONL file.
    """

    corpus_path: Path
    source_id: str = ""
    _texts: list[str] = field(default_factory=list, repr=False)
    _manifest: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, corpus_path: str | Path, verify_hash: bool = True
             ) -> "CorpusPool":
        corpus_path : Path = Path(corpus_path)
        if not corpus_path.exists():
            raise CorpusError(
                f"Corpus file not found: {corpus_path}. "
                f"Run scripts/build_corpus.py to build and pin the corpus."
            )

        manifest_path = corpus_path.with_suffix(
            corpus_path.suffix + ".manifest.json")
        manifest: dict = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if verify_hash and "sha256" in manifest:
                digest = hashlib.sha256(
                    corpus_path.read_bytes()).hexdigest()
                if digest != manifest["sha256"]:
                    raise CorpusError(
                        f"Corpus hash mismatch for {corpus_path.name}: "
                        f"manifest={manifest['sha256'][:12]}... "
                        f"actual={digest[:12]}... "
                        f"The corpus file has changed since it was pinned."
                    )

        texts: list[str] = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                texts.append(rec["text"])

        if len(texts) < 20:
            raise CorpusError(
                f"Corpus {corpus_path.name} has only {len(texts)} "
                f"paragraphs; refusing to use a pool this small."
            )

        pool = cls(corpus_path=corpus_path,
                   source_id=manifest.get("source_id", corpus_path.stem))
        pool._texts = texts
        pool._manifest = manifest
        return pool

    def paragraphs(self, rng: random.Random, k: int) -> list[str]:
        """k paragraphs, sampled WITHOUT replacement while k allows it,
        so distinct draws within a session use distinct paragraphs."""
        n = len(self._texts)
        if k <= n:
            idxs = rng.sample(range(n), k)
        else:
            idxs = [rng.randrange(n) for _ in range(k)]
        return [self._texts[i] for i in idxs]

    def manifest(self) -> dict:
        return {
            "source_id": self.source_id,
            "kind": "pinned_corpus",
            "path": str(self.corpus_path),
            "n_paragraphs": len(self._texts),
            **{k: v for k, v in self._manifest.items()
               if k in ("source", "license", "sha256", "built_at",
                        "builder_version")},
        }


# Active-source registry
_ACTIVE: Optional[CorpusPool] = None


def configure_content_source(source: CorpusPool) -> None:
    """Override the default corpus (e.g. pin a different version)."""
    global _ACTIVE
    _ACTIVE = source


def get_content_source() -> CorpusPool:
    """The active corpus, lazily loading the default pin if unset."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = CorpusPool.load(DEFAULT_CORPUS_PATH)
    return _ACTIVE


def active_content_manifest() -> dict:
    return get_content_source().manifest()


# Sentence utilities and the exact top-up wrapper

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences_of(paragraphs: list[str]) -> list[str]:
    out: list[str] = []
    for p in paragraphs:
        out.extend(s.strip() for s in _SENT_SPLIT.split(p) if s.strip())
    return out


def top_up_to_tokens(
    text: str,
    target_tokens: int,
    tokenizer,                       # TokenizerWrapper (count_tokens,
    filler_paragraphs: list[str],    #                   truncate_to_tokens)
    rng: random.Random,
    tolerance: float = 0.05,
    exact: bool = False,
) -> str:
    if target_tokens <= 0:
        return ""

    actual = tokenizer.count_tokens(text)

    if actual < target_tokens:
        fillers = sentences_of(filler_paragraphs)
        rng.shuffle(fillers)
        parts = [text] if text else []
        i = 0
        while actual < target_tokens and fillers:
            take = max(1, (target_tokens - actual) // 12)
            slice_ = [fillers[(i + j) % len(fillers)] for j in range(take)]
            i += take
            parts.append(" ".join(slice_))
            text = " ".join(parts)
            actual = tokenizer.count_tokens(text)
            if i > 10 * len(fillers) + 100:      # safety valve
                break

    if exact:
        if tokenizer.count_tokens(text) < target_tokens:
            raise RuntimeError(
                f"top_up_to_tokens could not reach {target_tokens} tokens "
            )
        return tokenizer.truncate_to_tokens(text, target_tokens)

    actual = tokenizer.count_tokens(text)
    if actual > int(target_tokens * (1 + tolerance)):
        text = tokenizer.truncate_to_tokens(text, target_tokens)
    return text