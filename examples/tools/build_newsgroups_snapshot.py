#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Licensed under the MIT License. See the LICENSE file in the repository root.
"""Rebuild the checked-in 20 Newsgroups snapshot used by the scikit-learn example.

The example itself never touches the network: it reads the CSV this script
writes. Run this only when you want to regenerate or extend the snapshot.

    python3 examples/tools/build_newsgroups_snapshot.py

Source: the 20 Newsgroups text dataset, fetched through
``sklearn.datasets.fetch_20newsgroups`` (~14 MB on first call, cached in
``~/scikit_learn_data``). Usenet headers, quoted replies, and signature blocks
are stripped, which is the standard protection against the metadata leakage
that makes this benchmark look easier than it is. Posts that still contain an
email address or a telephone number are left out of the candidate pool, so the
repository stores no such contact details; the kept posts are not edited. A post
that repeats is kept once, and a text posted to both groups is dropped because it
has no single label.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

CATEGORIES = {
    "comp.sys.ibm.pc.hardware": "pc",
    "comp.sys.mac.hardware": "mac",
}
POSTS_PER_CLASS = 60
DISCOVERY_PER_CLASS = 4
# Email addresses (also user@node forms without a dot) and telephone numbers:
# international, North American, toll-free, and numbers introduced by a label.
CONTACT = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*"
    r"|\+\d{1,3}[ .-]?\(?\d+\)?[ .\d/-]{5,}\d"
    r"|\(\d{3}\)\s?\d{3}[ .-]\d{4}"
    r"|(?<!\d)(?:1[ .-])?\d{3}[ .-]\d{3}[ .-]\d{4}(?!\d)"
    r"|\b(?:tel|phone|fax|voice)\b\.?:? ?[+(\d]",
    re.IGNORECASE,
)
MIN_CHARS = 400
MAX_CHARS = 1200

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "newsgroups_hardware" / "posts.csv"


def normalize(text: str) -> str:
    """Collapse Usenet whitespace and cut the post to a bounded length."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= MAX_CHARS:
        return collapsed
    truncated = collapsed[:MAX_CHARS]
    cut = truncated.rfind(" ")
    return truncated[:cut] if cut > MAX_CHARS // 2 else truncated


def build() -> pd.DataFrame:
    pools = {}
    for category, label in CATEGORIES.items():
        bundle = fetch_20newsgroups(
            subset="all",
            categories=[category],
            remove=("headers", "footers", "quotes"),
            shuffle=False,
        )
        pool = {}
        for raw in bundle.data:
            text = normalize(raw)
            if len(text) < MIN_CHARS or CONTACT.search(text):
                continue
            # Keyed by content hash, so a repeated post collapses to one entry.
            pool[hashlib.sha1(text.encode("utf-8")).hexdigest()] = text
        pools[label] = pool

    # A text posted to both groups has no single label, so neither keeps it.
    shared = set.intersection(*(set(pool) for pool in pools.values()))

    rows = []
    for category, label in CATEGORIES.items():
        # Sorting by content hash gives a stable, seed-free sample: the same
        # posts are selected on every machine and every sklearn version.
        candidates = sorted((d, t) for d, t in pools[label].items() if d not in shared)
        needed = DISCOVERY_PER_CLASS + POSTS_PER_CLASS
        selected = candidates[:needed]
        if len(selected) < needed:
            raise RuntimeError(
                f"{category}: only {len(selected)} posts passed the length filter, needed {needed}"
            )
        # The first slice feeds schema discovery and is held out of the
        # modelling set, so no post ever informs both the schema and a fold.
        for position, (digest, text) in enumerate(selected):
            rows.append(
                {
                    "post_id": f"{label}_{digest[:8]}",
                    "label": label,
                    "split": "discovery" if position < DISCOVERY_PER_CLASS else "model",
                    "text": text,
                }
            )

    df = pd.DataFrame(rows).sort_values("post_id", ignore_index=True)
    return df


def main() -> int:
    df = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {len(df)} posts to {OUTPUT_PATH}")
    print(pd.crosstab(df["split"], df["label"]).to_string())
    print(f"median length: {int(df['text'].str.len().median())} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
