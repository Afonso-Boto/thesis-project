from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

BUILDER_VERSION = "1.0"

_ARTIFACTS = [(" @-@ ", "-"), (" @.@ ", "."), (" @,@ ", ",")]
_HEADING = re.compile(r"^\s*=+.*=+\s*$")


def clean_line(line: str) -> str:
    for a, b in _ARTIFACTS:
        line = line.replace(a, b)
    return line.strip()


def letter_fraction(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for c in text if c.isalpha() or c.isspace())
    return letters / len(text)


def load_wikitext_lines() -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The one-time corpus build needs the `datasets` package:\n"
            "    pip install datasets\n"
            "(Only this build script needs it; the generator loads the "
            "pinned JSONL offline.)"
        )
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train")
    return (row["text"] for row in ds)  # generator; large split


def main() -> None:
    p = argparse.ArgumentParser(description="Build pinned prose corpus")
    p.add_argument("--name", default="wikitext103_prose_v1")
    p.add_argument("--n-paragraphs", type=int, default=2000)
    p.add_argument("--min-chars", type=int, default=300)
    p.add_argument("--max-chars", type=int, default=1400)
    p.add_argument("--min-letter-frac", type=float, default=0.82)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default="data/corpus")
    args = p.parse_args()

    out_dir = _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / f"{args.name}.jsonl"
    manifest_path = out_dir / f"{args.name}.jsonl.manifest.json"

    print("Loading WikiText-103 (train split)...")
    seen: set = set()
    kept: list[str] = []
    for raw in load_wikitext_lines():
        line = clean_line(raw)
        if not line or _HEADING.match(line):
            continue
        if not (args.min_chars <= len(line) <= args.max_chars):
            continue
        if letter_fraction(line) < args.min_letter_frac:
            continue
        if not line.endswith((".", "!", "?", '"')):
            continue
        h = hashlib.md5(line.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        kept.append(line)
        # collect generously, then subsample deterministically
        if len(kept) >= args.n_paragraphs * 20:
            break

    if len(kept) < args.n_paragraphs:
        print(f"WARNING: only {len(kept)} paragraphs passed filters "
              f"(requested {args.n_paragraphs}); writing what we have.")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    selected = kept[: args.n_paragraphs]

    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(selected):
            f.write(json.dumps({"id": f"wt103_{i:05d}", "text": text},
                               ensure_ascii=False) + "\n")

    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    manifest = {
        "source_id": args.name,
        "source": "WikiText-103 raw (Salesforce/wikitext, "
                  "wikitext-103-raw-v1, train split)",
        "license": "CC BY-SA 3.0 (derived from English Wikipedia)",
        "builder_version": BUILDER_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_letter_frac": args.min_letter_frac,
            "ends_with_sentence_punct": True,
            "headings_dropped": True,
            "artifacts_unescaped": True,
            "dedup": "exact",
        },
        "seed": args.seed,
        "n_paragraphs": len(selected),
        "sha256": digest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2),
                             encoding="utf-8")

    chars = [len(t) for t in selected]
    print(f"\nCorpus written: {corpus_path}")
    print(f"  paragraphs: {len(selected)}")
    print(f"  chars/paragraph: min {min(chars)}, median "
          f"{sorted(chars)[len(chars)//2]}, max {max(chars)}")
    print(f"  sha256: {digest[:16]}...")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()