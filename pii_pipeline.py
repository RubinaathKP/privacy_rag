"""
pii_pipeline.py - Steps A & B: Synthetic PII Injection and Hybrid Detection Pipeline

Part of the Privacy-Preserving Retrieval-Augmented Generation (RAG) Framework.
Affiliation: Technology Innovation Hub (IDEAS-TIH), Indian Statistical Institute (ISI), Kolkata.
Mentorship: Dr. Chandan Biswas.
Author / Student Researcher: RUBINAATH KP.

Overview:
---------
This module implements the end-to-end local data generation and detection pipeline
for Phase 1:
1. Step A (Synthetic Injection): Streams real passages from the MS MARCO dataset
   ('mteb/msmarco'), segments them into natural sentences using spaCy, randomly
   samples k in [0, 3] PII types (NAME, PHONE, EMAIL, ID, DATE, LOC), generates
   synthetic values via Faker, inserts natural carrier sentences at sentence boundaries,
   and records exact character offsets.
2. Step B (Hybrid Detection & Masking): Combines high-precision regular expressions
   (for EMAIL, PHONE, ID, DATE) with statistical Named Entity Recognition via spaCy's
   'en_core_web_sm' (for NAME, LOC). Applies heuristic filters (determiner trimming,
   mojibake suppression, URL exclusion, case sensitivity) and resolves overlapping
   spans using strict priority: Injected (0) > Regex (1) > spaCy (2).
3. Output Generation: Generates document-level presence labels and token-level binary
   masks (0 = non-PII, 1 = PII) for every token, outputting structured records to
   'data/pii_marco.jsonl'.
4. Sanity Verification: Computes and prints the Synthetic PII Recovery Table as an
   automated smoke test to ensure pipeline mechanics function properly.

Usage:
------
    python pii_pipeline.py [N_ROWS]
    Example: python pii_pipeline.py 1000
"""

import os
import re
import sys
import json
import random
from typing import List, Dict, Any, Tuple, Optional
import spacy
from faker import Faker

# --- 2. Global configuration ---
SEED = 42
random.seed(SEED)
Faker.seed(SEED)
fake = Faker()

DATASET = "mteb/msmarco"
N = 10
OUTPUT_PATH = "data/pii_marco.jsonl"
LABELER_VERSION = "step-ab-v1"
TYPES = ["NAME", "PHONE", "EMAIL", "ID", "DATE", "LOC"]

nlp = spacy.load("en_core_web_sm")

# --- 3. Step A: Synthetic PII injection ---

GENERATORS = {
    "NAME": fake.name,
    "PHONE": fake.phone_number,
    "EMAIL": fake.email,
    "ID": lambda: fake.bothify("??-#####").upper(),
    "DATE": lambda: fake.date_this_decade().strftime("%d-%m-%Y"),
    "LOC": fake.city,
}

TEMPLATES = {
    "NAME": [
        "{v} was mentioned",
        "a message from {v} arrived",
        "the file lists {v}",
        "{v} asked about this",
    ],
    "PHONE": [
        "call {v} for help",
        "the number on file is {v}",
        "reach the desk at {v}",
        "a callback to {v} was requested",
    ],
    "EMAIL": [
        "send questions to {v}",
        "the reply went to {v}",
        "the address on record is {v}",
    ],
    "ID": [
        "reference {v} was used",
        "ticket {v} was opened",
        "the case is filed under {v}",
        "{v} appears in the log",
    ],
    "DATE": [
        "the event took place on {v}",
        "the next review is {v}",
        "the deadline was set for {v}",
    ],
    "LOC": [
        "the meeting was held in {v}",
        "shipping went to {v}",
        "the office is located in {v}",
    ],
}

OPENERS = ["", "", "Also, ", "Note: ", "In addition, "]
CLOSERS = [".", ".", "!"]

# --- 4. Step B: Detection Patterns and Rules ---

MONTHS_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

REGEX_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "ID": re.compile(
        r"\b[A-Z]{2,4}-\d{3,5}\b|"
        r"\b[a-zA-Z]{3,}\d{4,}\b|"
        r"(?<!\d)\b\d{5}(?:-\d{4})?\b(?!\d)"
    ),
    "DATE": re.compile(
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|"
        rf"\b{MONTHS_PATTERN}(?:/{MONTHS_PATTERN})?\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b|"
        rf"\b{MONTHS_PATTERN}(?:/{MONTHS_PATTERN})?\s+\d{4}\b|"
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|"
        r"(?<![a-zA-Z0-9])(?:1\d{3}|20\d{2})(?:[-–](?:1\d{3}|20\d{2}))?(?![a-zA-Z0-9])"
    ),
    "PHONE": re.compile(
        r"(?<![\w-])(?:\+\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)|\d{2,4})(?:[-.\s]?\d{2,4}){1,3}(?:\s?x\d{1,5})?(?![\w-])"
    ),
}

DATE_FULLMATCH = re.compile(r"^\d{2}-\d{2}-\d{4}$")
YEAR_RANGE_FULLMATCH = re.compile(r"^(?:1\d{3}|20\d{2})[-–](?:1\d{3}|20\d{2})$")

URL_REGEX = re.compile(
    r"(?:https?://\S+|www\.\S+|(?:[a-zA-Z0-9-]+\.)+(?:gov|com|org|net|edu)(?:/\S*)?)",
    re.IGNORECASE,
)

MOJIBAKE_CHARS = set("âÂÃ¦€™")
PHONE_SEPARATORS = set("-.\t \r\n()+")

PRIORITY = {"injected": 0, "regex": 1, "spacy": 2}


def valid_phone(value: str) -> bool:
    """Validate phone number string according to specification 4.2."""
    val = value.strip()
    # Reject if the whole value fully matches DATE pattern or year-range pattern
    if DATE_FULLMATCH.match(val) or YEAR_RANGE_FULLMATCH.match(val):
        return False

    # Strip any extension (everything from 'x', case-insensitive) before counting digits
    base_part = re.split(r"x", value, flags=re.IGNORECASE)[0]
    digits = [c for c in base_part if c.isdigit()]
    if not (7 <= len(digits) <= 15):
        return False

    # Require at least one separator in the full value
    if not any(c in PHONE_SEPARATORS for c in value):
        return False

    return True


def trim_entity(ent) -> Optional[Tuple[int, int, List[Any]]]:
    """
    Drop leading tokens that are determiners (pos_ == 'DET') or punctuation,
    and drop trailing punctuation. If nothing remains, return None.
    Returns (start_char, end_char, tokens).
    """
    tokens = list(ent)
    while tokens and (tokens[0].pos_ == "DET" or tokens[0].is_punct):
        tokens.pop(0)
    while tokens and tokens[-1].is_punct:
        tokens.pop(-1)

    if not tokens:
        return None

    start = tokens[0].idx
    end = tokens[-1].idx + len(tokens[-1].text)
    return start, end, tokens


def keep_spacy(etype: str, text: str, tokens: List[Any]) -> bool:
    """Check filtering conditions for spaCy candidate entity."""
    if "--" in text:
        return False
    if any(c in MOJIBAKE_CHARS for c in text):
        return False
    if not text or not text[0].isupper():
        return False
    if etype == "NAME" and not any(t.pos_ == "PROPN" for t in tokens):
        return False
    return True


def resolve(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort by priority (injected = 0, regex = 1, spacy = 2),
    then by longer length first, then by start.
    Keep a span only if it does not overlap any already-kept span.
    Return sorted by start.
    """
    sorted_spans = sorted(
        spans,
        key=lambda s: (
            PRIORITY.get(s.get("source", ""), 99),
            -(s["end"] - s["start"]),
            s["start"],
        ),
    )
    kept = []
    for span in sorted_spans:
        overlap = False
        for k in kept:
            if max(span["start"], k["start"]) < min(span["end"], k["end"]):
                overlap = True
                break
        if not overlap:
            kept.append(span)

    return sorted(kept, key=lambda s: (s["start"], s["end"]))


def detect(text: str, doc=None) -> List[Dict[str, Any]]:
    """
    Step B detection: regex + spaCy, trimmed, filtered, and resolved.
    Returns list of dicts: {start, end, type, source, text}.
    """
    spans: List[Dict[str, Any]] = []

    # 1. Regex detectors
    for etype, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            val = m.group()
            if etype == "PHONE" and not valid_phone(val):
                continue
            spans.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "type": etype,
                    "source": "regex",
                    "text": val,
                }
            )

    # 2. spaCy detectors
    if doc is None:
        doc = nlp(text)

    # Collect URLs to filter overlapping spaCy spans
    urls = [(m.start(), m.end()) for m in URL_REGEX.finditer(text)]

    label_map = {
        "PERSON": "NAME",
        "GPE": "LOC",
        "LOC": "LOC",
        "DATE": "DATE",
    }

    for ent in doc.ents:
        if ent.label_ not in label_map:
            continue
        etype = label_map[ent.label_]

        trimmed = trim_entity(ent)
        if trimmed is None:
            continue
        s, e, tokens = trimmed
        ent_text = text[s:e]

        # Overlaps URL?
        if any(max(s, us) < min(e, ue) for us, ue in urls):
            continue

        if not keep_spacy(etype, ent_text, tokens):
            continue

        spans.append(
            {
                "start": s,
                "end": e,
                "type": etype,
                "source": "spacy",
                "text": ent_text,
            }
        )

    resolved = resolve(spans)
    # Ensure text field matches text[start:end]
    for s in resolved:
        s["text"] = text[s["start"] : s["end"]]
    return resolved


def make_injection(etype: str) -> Dict[str, Any]:
    """
    3.4 make_injection(etype)
    Generate synthetic PII value and carrier sentence with exact offset.
    """
    value = GENERATORS[etype]()
    template = random.choice(TEMPLATES[etype])
    opener = random.choice(OPENERS)

    if opener == "" and template and template[0].isalpha():
        template_to_use = template[0].upper() + template[1:]
    else:
        template_to_use = template

    prefix = opener
    closer = random.choice(CLOSERS)

    sentence = prefix + template_to_use.format(v=value) + closer
    offset = len(prefix) + template_to_use.index("{v}")

    return {
        "type": etype,
        "value": value,
        "sentence": sentence,
        "offset": offset,
    }


def build_record(
    original: str, doc=None, k: Optional[int] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    3.5 build_record(original, doc)
    Build processed record with synthetic injections and mapped natural spans.
    Returns (record_dict, injected_spans_for_sanity).
    """
    if doc is None:
        doc = nlp(original)

    # 1. Sentence slices
    sents: List[Tuple[int, int]] = []
    for sent in doc.sents:
        if sent.text.strip():
            sents.append((sent.start_char, sent.end_char))

    if not sents:
        sents = [(0, len(original))]

    # 2. Pick k injections
    if k is None:
        k = random.choice([0, 1, 1, 2])

    injected_types_selected = random.sample(TYPES, k) if k > 0 else []

    # Assign pos
    injections_by_pos: Dict[int, List[Dict[str, Any]]] = {
        i: [] for i in range(len(sents) + 1)
    }
    for etype in injected_types_selected:
        inj = make_injection(etype)
        pos = random.randint(0, len(sents))
        injections_by_pos[pos].append(inj)

    # 3. Assemble chunks
    # List of tuples: (kind, item)
    # kind: "injection" (item: inj dict) or "original" (item: sent_idx)
    chunks: List[Tuple[str, Any]] = []
    for i in range(len(sents)):
        for inj in injections_by_pos[i]:
            chunks.append(("injection", inj))
        chunks.append(("original", i))
    for inj in injections_by_pos[len(sents)]:
        chunks.append(("injection", inj))

    processed_pieces: List[str] = []
    curr_offset = 0
    injected_spans: List[Dict[str, Any]] = []
    orig_sent_mapped: Dict[int, Tuple[int, int, int]] = {}  # sent_idx -> (chunk_start, orig_s, orig_e)

    for idx, (kind, item) in enumerate(chunks):
        if idx > 0:
            curr_offset += 1  # single space between chunks

        chunk_start = curr_offset
        if kind == "injection":
            inj = item
            chunk_text = inj["sentence"]
            start = chunk_start + inj["offset"]
            end = start + len(inj["value"])
            injected_spans.append(
                {
                    "start": start,
                    "end": end,
                    "type": inj["type"],
                    "source": "injected",
                    "text": inj["value"],
                    "sentence_start": chunk_start,
                    "sentence_end": chunk_start + len(chunk_text),
                }
            )
            processed_pieces.append(chunk_text)
            curr_offset += len(chunk_text)
        else:
            sent_idx = item
            orig_s, orig_e = sents[sent_idx]
            chunk_text = original[orig_s:orig_e]
            orig_sent_mapped[sent_idx] = (chunk_start, orig_s, orig_e)
            processed_pieces.append(chunk_text)
            curr_offset += len(chunk_text)

    processed_text = " ".join(processed_pieces)

    # 4. Map natural PII from original text
    natural_spans = detect(original, doc)
    mapped_natural_spans: List[Dict[str, Any]] = []

    for nat_span in natural_spans:
        ns = nat_span["start"]
        ne = nat_span["end"]
        # Find containing sentence
        containing_sent = None
        for sent_idx, (chunk_start, orig_s, orig_e) in orig_sent_mapped.items():
            if orig_s <= ns and ne <= orig_e:
                containing_sent = (chunk_start, orig_s, orig_e)
                break

        if containing_sent is None:
            # Span crosses sentence boundary or sentence not mapped; drop it
            continue

        chunk_start, orig_s, orig_e = containing_sent
        mapped_start = chunk_start + (ns - orig_s)
        mapped_end = chunk_start + (ne - orig_s)

        if processed_text[mapped_start:mapped_end] == nat_span["text"]:
            mapped_natural_spans.append(
                {
                    "start": mapped_start,
                    "end": mapped_end,
                    "type": nat_span["type"],
                    "source": nat_span["source"],
                    "text": nat_span["text"],
                }
            )

    # 5. Merge injected and mapped natural spans with resolve()
    all_spans = [
        {
            "start": s["start"],
            "end": s["end"],
            "type": s["type"],
            "source": s["source"],
            "text": s["text"],
        }
        for s in injected_spans
    ] + mapped_natural_spans

    final_spans = resolve(all_spans)
    # Ensure text matches processed_text
    for s in final_spans:
        s["text"] = processed_text[s["start"] : s["end"]]

    # 6. Document labels
    labels = {
        t: (1 if any(s["type"] == t for s in final_spans) else 0)
        for t in TYPES
    }

    # 7. Token mask using nlp.make_doc(text) (tokenizer only, no NER)
    proc_doc = nlp.make_doc(processed_text)
    tokens: List[str] = [tok.text for tok in proc_doc]
    binary_mask: List[int] = []

    for tok in proc_doc:
        tok_start = tok.idx
        tok_end = tok.idx + len(tok.text)
        overlaps = any(
            max(tok_start, s["start"]) < min(tok_end, s["end"])
            for s in final_spans
        )
        binary_mask.append(1 if overlaps else 0)

    injected_types_sorted = sorted(list(set(s["type"] for s in injected_spans)))

    record = {
        "original": original,
        "text": processed_text,
        "labels": labels,
        "spans": final_spans,
        "tokens": tokens,
        "binary_mask": binary_mask,
        "injected_types": injected_types_sorted,
        "labeler_version": LABELER_VERSION,
    }

    # Return record and injected_spans (cleaned without extra debug keys if any)
    sanity_injected = [
        {
            "start": s["start"],
            "end": s["end"],
            "type": s["type"],
            "source": s["source"],
            "text": s["text"],
            "sentence_start": s.get("sentence_start"),
            "sentence_end": s.get("sentence_end"),
        }
        for s in injected_spans
    ]
    return record, sanity_injected


def run(n: int = N, out_path: str = OUTPUT_PATH) -> None:
    """
    Stream n rows from dataset, build records, write JSONL, and print sanity recovery table.
    """
    from datasets import load_dataset

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Loading streaming dataset: {DATASET}...")
    ds = load_dataset(DATASET, "corpus", split="corpus", streaming=True)

    records: List[Dict[str, Any]] = []
    all_injected_spans: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []

    count = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, row in enumerate(ds):
            text = row.get("text", "")
            if not text or not text.strip():
                continue

            doc = nlp(text)
            record, injected = build_record(text, doc)
            record["id"] = str(row.get("_id", count))

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            all_injected_spans.append((record, injected))

            count += 1
            if count >= n:
                break

    print(f"Wrote {count} records to {out_path}.")

    # Sanity check: for each injected span, check if detect(record["text"]) returns a span of same type overlapping it
    counts_total: Dict[str, int] = {t: 0 for t in TYPES}
    counts_recovered: Dict[str, int] = {t: 0 for t in TYPES}

    for record, injected_list in all_injected_spans:
        detected = detect(record["text"])
        for inj in injected_list:
            t = inj["type"]
            counts_total[t] += 1
            # Check overlap with same type in detected
            rec = any(
                d["type"] == t
                and max(d["start"], inj["start"]) < min(d["end"], inj["end"])
                for d in detected
            )
            if rec:
                counts_recovered[t] += 1

    print("\n" + "=" * 50)
    print("SANITY CHECK: Synthetic PII Recovery Table")
    print("=" * 50)
    print(f"{'Type':<10} {'Total':<10} {'Recovered':<12} {'Recovery Rate':<15}")
    print("-" * 50)
    for t in TYPES:
        tot = counts_total[t]
        rec = counts_recovered[t]
        rate_str = f"{(rec / tot * 100):.1f}%" if tot > 0 else "N/A"
        print(f"{t:<10} {tot:<10} {rec:<12} {rate_str:<15}")
    print("-" * 50)
    tot_all = sum(counts_total.values())
    rec_all = sum(counts_recovered.values())
    rate_all = f"{(rec_all / tot_all * 100):.1f}%" if tot_all > 0 else "N/A"
    print(f"{'TOTAL':<10} {tot_all:<10} {rec_all:<12} {rate_all:<15}")
    print("=" * 50)
    print("Note: Sanity only: synthetic text is regular. Real recall needs evaluate_gold.\n")


def evaluate_gold(gold_path: str) -> Dict[str, Any]:
    """
    Evaluate detector on hand-labelled gold set (natural PII recall).
    Matching rule: predicted span is TP for a gold span if same type and overlapping ranges.
    Each predicted span matches at most one gold span.
    """
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    gold_records = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_records.append(json.loads(line))

    tp_counts: Dict[str, int] = {t: 0 for t in TYPES}
    fp_counts: Dict[str, int] = {t: 0 for t in TYPES}
    fn_counts: Dict[str, int] = {t: 0 for t in TYPES}

    for item in gold_records:
        text = item["text"]
        gold_spans = item.get("spans", [])
        pred_spans = detect(text)

        for t in TYPES:
            g_t = [g for g in gold_spans if g["type"] == t]
            p_t = [p for p in pred_spans if p["type"] == t]

            matched_g_indices = set()
            for p in p_t:
                match_found = False
                for g_idx, g in enumerate(g_t):
                    if g_idx in matched_g_indices:
                        continue
                    if max(p["start"], g["start"]) < min(p["end"], g["end"]):
                        match_found = True
                        matched_g_indices.add(g_idx)
                        break
                if match_found:
                    tp_counts[t] += 1
                else:
                    fp_counts[t] += 1

            fn_counts[t] += len(g_t) - len(matched_g_indices)

    print("\n" + "=" * 65)
    print("GOLD SET EVALUATION: Natural PII Recall & Precision")
    print("=" * 65)
    print(f"{'Type':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<14} {'Recall':<14}")
    print("-" * 65)

    metrics = {}
    for t in TYPES:
        tp = tp_counts[t]
        fp = fp_counts[t]
        fn = fn_counts[t]
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec_str = f"{prec:.2%}" if (tp + fp) > 0 else "0.00%"
        rec_str = f"{rec:.2%}" if (tp + fn) > 0 else "0.00%"
        print(f"{t:<10} {tp:<8} {fp:<8} {fn:<8} {prec_str:<14} {rec_str:<14}")
        metrics[t] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}

    tot_tp = sum(tp_counts.values())
    tot_fp = sum(fp_counts.values())
    tot_fn = sum(fn_counts.values())
    tot_prec = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) > 0 else 0.0
    tot_rec = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) > 0 else 0.0
    tot_prec_str = f"{tot_prec:.2%}" if (tot_tp + tot_fp) > 0 else "0.00%"
    tot_rec_str = f"{tot_rec:.2%}" if (tot_tp + tot_fn) > 0 else "0.00%"
    print("-" * 65)
    print(f"{'TOTAL':<10} {tot_tp:<8} {tot_fp:<8} {tot_fn:<8} {tot_prec_str:<14} {tot_rec_str:<14}")
    print("=" * 65)
    print("Note: Precision is reported as a utility cost (over-labelling is intentional).")
    print("      Recall is the key privacy metric.\n")

    metrics["TOTAL"] = {
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "precision": tot_prec,
        "recall": tot_rec,
    }
    return metrics


if __name__ == "__main__":
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else N
    run(n=n_arg)
