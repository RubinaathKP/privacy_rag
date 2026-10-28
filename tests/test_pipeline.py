"""
test_pipeline.py - Deterministic Unit Test Suite for Step A, Step B, and Gold Pipeline

Part of the Privacy-Preserving Retrieval-Augmented Generation (RAG) Framework.
Affiliation: Technology Innovation Hub (IDEAS-TIH), Indian Statistical Institute (ISI), Kolkata.
Mentorship: Dr. Chandan Biswas.
Author / Student Researcher: RUBINAATH KP.

Overview:
---------
This module provides 9 comprehensive, deterministic unit tests validating the core
invariants of the Privacy-Preserving RAG pipeline specified in the SOW (Section 7):
1. Injected span exact character-offset matching (record['text'][start:end] == span['text']).
2. Pairwise non-overlap of injected synthetic entities.
3. Sentence boundary punctuation adherence (all injected sentences terminate with '.', '!', or '?').
4. Token mask dimensional parity (len(binary_mask) == len(tokens)) and binary validity in {0, 1}.
5. Document label correctness (labels[t] == 1 iff a span of type t exists).
6. Phone validation logic (valid separator/digit constraints and rejection of date patterns).
7. URL false-positive suppression (detect returns no spans overlapping URLs).
8. Clean record baseline behavior (k=0 produces no injected spans or injected sources).
9. Substring to exact character-offset conversion in gold benchmark mapping.

Usage:
------
    pytest tests/
"""

import os
import pytest
from pii_pipeline import (
    nlp,
    TYPES,
    URL_REGEX,
    build_record,
    detect,
    valid_phone,
)
from gold import to_offsets


def test_1_injected_spans_exact_match():
    """1. For every injected span in a built record, record['text'][start:end] == span['text']."""
    synthetic_text = (
        "The rapid brown fox jumps over the sleeping dog. "
        "Artificial intelligence systems are changing research workflows."
    )
    doc = nlp(synthetic_text)
    record, injected = build_record(synthetic_text, doc, k=2)
    assert len(injected) > 0, "Expected at least 1 injected span when k=2"
    for span in injected:
        s, e = span["start"], span["end"]
        assert record["text"][s:e] == span["text"]


def test_2_injected_spans_do_not_overlap():
    """2. Injected spans do not overlap each other."""
    synthetic_text = (
        "Sentence one is here. Sentence two follows it closely. "
        "Sentence three concludes the small passage."
    )
    doc = nlp(synthetic_text)
    record, injected = build_record(synthetic_text, doc, k=2)
    for i in range(len(injected)):
        for j in range(i + 1, len(injected)):
            s1, e1 = injected[i]["start"], injected[i]["end"]
            s2, e2 = injected[j]["start"], injected[j]["end"]
            assert max(s1, s2) >= min(e1, e2), (
                f"Injected spans overlap: {injected[i]} and {injected[j]}"
            )


def test_3_injected_sentences_end_with_punctuation():
    """3. Every injected sentence ends with '.', '!', or '?' in the processed text."""
    synthetic_text = "Primary sentence is here. Secondary sentence is there."
    doc = nlp(synthetic_text)
    record, injected = build_record(synthetic_text, doc, k=2)
    for span in injected:
        sent_s = span["sentence_start"]
        sent_e = span["sentence_end"]
        injected_sentence = record["text"][sent_s:sent_e]
        assert injected_sentence[-1] in {".", "!", "?"}, (
            f"Injected sentence does not end with punctuation: '{injected_sentence}'"
        )


def test_4_binary_mask_length_and_values():
    """4. len(binary_mask) == len(tokens), and every mask value is 0 or 1."""
    synthetic_text = "Alice lived in London in 2020. Her email was alice@example.com."
    doc = nlp(synthetic_text)
    record, _ = build_record(synthetic_text, doc, k=1)
    assert len(record["binary_mask"]) == len(record["tokens"])
    for val in record["binary_mask"]:
        assert val in (0, 1)


def test_5_document_labels_correspond_to_spans():
    """5. labels[t] == 1 if and only if some span of type t exists."""
    synthetic_text = "Contact support regarding ticket AB-12345 in Berlin."
    doc = nlp(synthetic_text)
    record, _ = build_record(synthetic_text, doc, k=1)
    for t in TYPES:
        has_span = any(s["type"] == t for s in record["spans"])
        assert record["labels"][t] == (1 if has_span else 0)


def test_6_valid_phone():
    """6. valid_phone('21-10-2021') is False. valid_phone('+1-210-343-3218x1960') is True."""
    assert valid_phone("21-10-2021") is False
    assert valid_phone("+1-210-343-3218x1960") is True


def test_7_detect_no_url_overlapping_spans():
    """7. detect returns no URL-overlapping spans for 'see http://www.cfo.doe.gov/me70/history'."""
    url_text = "see http://www.cfo.doe.gov/me70/history"
    detected = detect(url_text)
    url_match = URL_REGEX.search(url_text)
    assert url_match is not None
    us, ue = url_match.start(), url_match.end()
    for s in detected:
        overlap = max(s["start"], us) < min(s["end"], ue)
        assert not overlap, f"Detected span {s} overlaps URL {url_match.group()}"


def test_8_clean_record_k_zero():
    """8. A clean record (k = 0) has injected_types == [] and no source == 'injected' spans."""
    synthetic_text = "This is a clean synthetic document without any synthetic injection."
    doc = nlp(synthetic_text)
    record, injected = build_record(synthetic_text, doc, k=0)
    assert record["injected_types"] == []
    assert len(injected) == 0
    assert not any(s["source"] == "injected" for s in record["spans"])


def test_9_to_offsets_known_string():
    """9. to_offsets on a known string returns offsets where text[start:end] == value."""
    text = "John Doe visited Paris on 12-05-2023. Contact him at john@example.com."
    items = [
        {"text": "John Doe", "type": "NAME"},
        {"text": "Paris", "type": "LOC"},
        {"text": "12-05-2023", "type": "DATE"},
        {"text": "john@example.com", "type": "EMAIL"},
    ]
    spans = to_offsets(text, items)
    assert len(spans) == 4
    for s in spans:
        val = text[s["start"] : s["end"]]
        assert val == s["text"]
