"""
gold.py - Gold Standard Benchmark Generation, Ingestion, and Evaluation CLI

Part of the Privacy-Preserving Retrieval-Augmented Generation (RAG) Framework.
Affiliation: Technology Innovation Hub (IDEAS-TIH), Indian Statistical Institute (ISI), Kolkata.
Mentorship: Dr. Chandan Biswas.
Author / Student Researcher: RUBINAATH KP.

Overview:
---------
This module manages the gold standard evaluation corpus (75 clean MS MARCO passages,
272 human-verified ground-truth spans) used to measure the real-world privacy recall
and utility precision of the detection pipeline:
1. 'originals': Streams unmodified passages from MS MARCO starting at row index SKIP=1000,
   guaranteeing strict non-overlapping separation from the dev set (rows 0-999).
2. 'add_raw': Ingests raw LLM-assisted candidate extractions generated using the standardized
   prompt in 'prompts/gemini_gold_prompt.txt'.
3. 'build': Maps raw entity substrings to exact character-level coordinate offsets
   [start, end] in the original text, logging discrepancies for human-in-the-loop review.
4. 'evaluate': Evaluates the detection pipeline ('pii_pipeline.detect') against
   'data/gold_natural.jsonl', computing True Positives, False Positives, False Negatives,
   utility precision, and privacy recall across all 6 PII types.

Commands:
---------
    python gold.py originals   : Stream clean MS MARCO gold passages to data/originals.jsonl
    python gold.py add_raw     : Ingest raw JSON annotation items for a passage
    python gold.py build       : Build character-offset gold spans in data/gold_natural.jsonl
    python gold.py evaluate    : Evaluate detector recall and precision against gold set
"""

import os
import re
import sys
import json
import itertools
from typing import List, Dict, Any, Optional, Set

from pii_pipeline import DATASET, evaluate_gold

# --- Constants ---
SKIP = 1000
N_GOLD = 75
ORIG_PATH = "data/originals.jsonl"
RAW_PATH = "data/gemini_raw.jsonl"
GOLD_PATH = "data/gold_natural.jsonl"
TYPES: Set[str] = {"NAME", "LOC", "DATE", "PHONE", "EMAIL", "ID"}


def to_offsets(
    text: str,
    items: List[Dict[str, Any]],
    problems: Optional[List[str]] = None,
    passage_id: Any = None,
) -> List[Dict[str, Any]]:
    """
    Convert list of raw items [{"text": ..., "type": ...}] into character-offset spans.
    Finds every occurrence of the string, deduplicates by (start, end, type), and sorts by start.
    Logs items with bad types, empty text, or not found in passage.
    """
    if problems is None:
        problems = []

    spans: List[Dict[str, Any]] = []
    for item in items:
        val = item.get("text", "")
        t = item.get("type", "")
        if not val or not str(val).strip():
            problems.append(f"Passage {passage_id}: empty text item")
            continue
        val = str(val).strip()
        if t not in TYPES:
            problems.append(f"Passage {passage_id}: bad type '{t}' for item '{val}'")
            continue

        matches = list(re.finditer(re.escape(val), text))
        if not matches:
            problems.append(
                f"Passage {passage_id}: item '{val}' (type {t}) not found in text"
            )
            continue

        for m in matches:
            spans.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "type": t,
                    "text": text[m.start() : m.end()],
                }
            )

    # Deduplicate by (start, end, type)
    unique_spans: List[Dict[str, Any]] = []
    seen = set()
    for s in spans:
        key = (s["start"], s["end"], s["type"])
        if key not in seen:
            seen.add(key)
            unique_spans.append(s)

    return sorted(unique_spans, key=lambda s: (s["start"], s["end"]))


def make_originals() -> None:
    """Stream rows SKIP to SKIP + N_GOLD from MS MARCO corpus and write clean passages."""
    from datasets import load_dataset

    os.makedirs(os.path.dirname(ORIG_PATH) or ".", exist_ok=True)
    print(
        f"Loading {DATASET} streaming (skipping first {SKIP} rows, taking {N_GOLD} non-empty)..."
    )
    ds = load_dataset(DATASET, "corpus", split="corpus", streaming=True)

    stream = itertools.islice(ds, SKIP, None)
    count = 0
    with open(ORIG_PATH, "w", encoding="utf-8") as f:
        for row in stream:
            text = row.get("text", "")
            _id = row.get("_id", count)
            if not text or not text.strip():
                continue
            f.write(
                json.dumps({"id": str(_id), "text": text}, ensure_ascii=False) + "\n"
            )
            count += 1
            if count >= N_GOLD:
                break

    print(f"Wrote {count} original passages to {ORIG_PATH}")


def add_raw(passage_id: Any, pasted_output: str) -> None:
    """
    Append raw LLM json output for passage_id to RAW_PATH.
    Strips markdown code fences (```json or ```) and whitespace.
    """
    os.makedirs(os.path.dirname(RAW_PATH) or ".", exist_ok=True)
    raw = pasted_output.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    items = json.loads(raw)
    with open(RAW_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps({"id": str(passage_id), "items": items}, ensure_ascii=False)
            + "\n"
        )
    print(f"Added raw output for passage {passage_id} ({len(items)} items)")


def build_gold() -> None:
    """
    Load originals and raw outputs, keep the most recent raw output per id,
    convert items to offsets, and write GOLD_PATH.
    """
    if not os.path.exists(ORIG_PATH):
        print(f"Originals file not found at {ORIG_PATH}. Run 'python gold.py originals' first.")
        return

    originals = []
    with open(ORIG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                originals.append(json.loads(line))

    raw_by_id: Dict[str, List[Dict[str, Any]]] = {}
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    raw_by_id[str(item["id"])] = item.get("items", [])

    problems: List[str] = []
    gold_records: List[Dict[str, Any]] = []

    for orig in originals:
        pid = str(orig["id"])
        text = orig["text"]
        items = raw_by_id.get(pid, [])
        spans = to_offsets(text, items, problems, pid)
        gold_records.append({"id": pid, "text": text, "spans": spans})

    os.makedirs(os.path.dirname(GOLD_PATH) or ".", exist_ok=True)
    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        for rec in gold_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Built gold set with {len(gold_records)} passages -> {GOLD_PATH}")
    if problems:
        print(f"\n[!] Logged {len(problems)} problem(s) to fix by hand:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No problems logged during offset mapping.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gold.py <command>")
        print("Commands:")
        print("  originals : Stream clean gold passages from MS MARCO")
        print("  build     : Convert raw Gemini outputs to character-offset gold spans")
        print("  evaluate  : Evaluate detector recall and precision against gold set")
        print("\nNote: add_raw(passage_id, pasted_output) is called interactively from Python.")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "originals":
        make_originals()
    elif cmd == "add_raw":
        if len(sys.argv) < 3:
            print("Usage: python gold.py add_raw <passage_id> [json_string]")
            print("Or pipe/paste JSON directly into standard input.")
            sys.exit(1)
        pid = sys.argv[2]
        if len(sys.argv) >= 4:
            raw_str = " ".join(sys.argv[3:])
        else:
            print(f"Paste Gemini JSON output for passage {pid} (Ctrl+Z and Enter on Windows to finish):")
            raw_str = sys.stdin.read()
        add_raw(pid, raw_str)
    elif cmd == "build":
        build_gold()
    elif cmd == "evaluate":
        evaluate_gold(GOLD_PATH)
    else:
        print(f"Unknown command: {cmd}")
        print("Valid commands: originals, add_raw, build, evaluate")
        sys.exit(1)
