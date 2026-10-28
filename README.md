# Privacy-Preserving Retrieval-Augmented Generation (Phase 1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-9%20passed-brightgreen.svg)](tests/test_pipeline.py)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Affiliation](https://img.shields.io/badge/IDEAS--TIH-ISI%20Kolkata-red.svg)](https://www.isical.ac.in/)

**Research Project:** Privacy-Preserving Retrieval-Augmented Generation (RAG) — Phase 1: Synthetic Data Injection, Hybrid Detection & Gold Standard Evaluation  
**Host Institution:** Technology Innovation Hub (IDEAS-TIH), Indian Statistical Institute (ISI), Kolkata  
**Mentor:** Dr. Chandan Biswas  
**Student Researcher:** RUBINAATH KP  
**Version:** `step-ab-v1`  

---

## 1. Overview

This repository contains the implementation of **Phase 1 (Steps A and B)** of the Privacy-Preserving RAG framework. The project establishes an end-to-end, local, reproducible pipeline on the MS MARCO dataset (`mteb/msmarco`) to:
1. **Inject Controlled Synthetic PII (Step A)**: Augments streaming text with synthetic PII carrier sentences at sentence boundaries with exact character-offset mathematical tracking.
2. **Perform Hybrid Detection & Masking (Step B)**: Detects deterministic PII via regex and open-vocabulary entities via spaCy (`en_core_web_sm`), generates document-level presence labels, and produces token-level binary masks (`0` = non-PII, `1` = PII).
3. **Benchmark Natural PII (Gold Set Evaluation)**: Evaluates real-world recall and utility precision on 75 clean, unmodified MS MARCO passages containing 272 human-in-the-loop verified ground-truth spans, reporting Wilson score 95% confidence intervals.
4. **Produce Model Training Data**: Generates 1,000 processed records (`data/pii_marco.jsonl`) to serve as training and fine-tuning data for Phase 2 neural token classifiers (DeBERTa / RoBERTa).

---

## 2. Repository Structure

```
privacy_rag/
├── data/
│   ├── pii_marco.jsonl       # 1,000 processed dev records with token binary masks (rows 0-999)
│   ├── originals.jsonl       # 75 clean, unmodified gold passages from MS MARCO (rows 1000+)
│   ├── gemini_raw.jsonl      # Raw LLM candidate extractions per passage
│   └── gold_natural.jsonl    # 75 human-verified gold records (272 ground-truth spans)
├── prompts/
│   └── gemini_gold_prompt.txt # Standardized prompt for Section 8 LLM-assisted annotation
├── tests/
│   └── test_pipeline.py      # 9 deterministic unit tests covering pipeline invariants
├── .gitignore                # Git exclusions (caches, virtual environments, OS files)
├── gold.py                   # Gold-set streaming, ingestion, and evaluation CLI
├── pii_pipeline.py           # Core Step A & B injection, detection, and masking pipeline
├── README.md                 # Project documentation and reproduction guide
└── requirements.txt          # Python dependencies (datasets, spacy, faker, pytest)
```

---

## 3. Installation

Clone this repository and set up a Python 3.10+ environment:

```bash
# Clone the repository
git clone https://github.com/RubinaathKP/privacy_rag.git
cd privacy_rag

# Install dependencies
pip install -r requirements.txt

# Download the lightweight English spaCy model
python -m spacy download en_core_web_sm
```

### Dependencies
- `datasets` (Hugging Face streaming access for MS MARCO)
- `spacy` (tokenization, sentence segmentation, named entity recognition)
- `faker` (synthetic PII value generation)
- `pytest` (deterministic unit test suite)

---

## 4. Run Order & Execution

Follow this exact sequence to run and verify the entire pipeline:

### Step 1: Run Deterministic Unit Tests
Ensure all 9 invariant tests pass before pipeline execution:
```bash
pytest tests/
```
*Expected: `9 passed in ~2.5s`.*

### Step 2: Run Dev Pipeline (`pii_pipeline.py`)
Processes 1,000 streaming passages from MS MARCO (rows 0 to 999), injects synthetic PII, computes token masks, writes `data/pii_marco.jsonl`, and prints the synthetic sanity recovery table:
```bash
python pii_pipeline.py 1000
```

### Step 3: Stream Clean Gold Originals (`gold.py originals`)
Streams clean passages starting at row index 1000 without modification:
```bash
python gold.py originals
```
*Writes clean texts to `data/originals.jsonl`.*

### Step 4: Evaluate Natural Recall on Gold Benchmark (`gold.py evaluate`)
Evaluates the hybrid detector against the 75 clean, human-verified passages (272 natural spans):
```bash
python gold.py evaluate
```

---

## 5. Evaluation Results

### 5.1 Real-World Natural Gold Benchmark (75 Passages, 272 Ground-Truth Spans)

Evaluated against 100% natural, unmodified MS MARCO web search passages with Wilson score 95% confidence intervals ($\alpha = 0.05$):

| PII Category | Gold Support | Detected (TP) | False Pos (FP) | False Neg (FN) | Precision (Utility Cost) | 95% CI (Precision) | Recall (Privacy Metric) | 95% CI (Recall) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NAME** | 93 | 75 | 29 | 18 | 72.12% | [62.8%, 79.8%] | **80.65%** | [71.5%, 87.4%] |
| **PHONE** | 5 | 5 | 0 | 0 | 100.00% | [56.6%, 100.0%] | **100.00%** | [56.6%, 100.0%] |
| **EMAIL** | 3 | 3 | 0 | 0 | 100.00% | [43.9%, 100.0%] | **100.00%** | [43.9%, 100.0%] |
| **ID** | 4 | 4 | 0 | 0 | 100.00% | [51.0%, 100.0%] | **100.00%** | [51.0%, 100.0%] |
| **DATE** | 51 | 51 | 6 | 0 | 89.47% | [78.9%, 95.1%] | **100.00%** | [93.0%, 100.0%] |
| **LOC** | 116 | 72 | 3 | 44 | 96.00% | [88.9%, 98.6%] | **62.07%** | [53.0%, 70.4%] |
| **TOTAL** | **272** | **210** | **38** | **62** | **84.68%** | **[79.7%, 88.6%]** | **77.21%** | **[71.9%, 81.8%]** |

#### Key Empirical Insights:
- **Zero Privacy Leaks on Contact Data**: `PHONE` (5/5), `EMAIL` (3/3), `ID` (4/4), and `DATE` (51/51) achieved **100.0% Recall**.
- **Utility Cost vs. Privacy**: Precision on natural text is **84.68%**. In a privacy-first posture, over-labelling of dates (e.g. historical years) and locations is an intentional utility cost to prevent privacy leaks.
- **Empirical Isolation of the 62 False Negatives**: 100% of all missed spans come from `LOC` (44 misses) and `NAME` (18 misses). This stems from the natural capacity limit of lightweight statistical NER (`en_core_web_sm`) on compound nouns, all-caps text, and mononyms. This empirically establishes the exact baseline to be improved by Phase 2 neural fine-tuning.

---

### 5.2 Synthetic PII Recovery Smoke Test (N = 1,000 Dev Records)

Sanity test verifying that the detector can recover regular synthetic carrier sentences injected into the dev set:

| Type | Total Injected | Recovered | Recovery Rate |
| :--- | :--- | :--- | :--- |
| **NAME** | 148 | 139 | 93.9% |
| **PHONE** | 161 | 149 | 92.5% |
| **EMAIL** | 163 | 163 | 100.0% |
| **ID** | 167 | 167 | 100.0% |
| **DATE** | 177 | 177 | 100.0% |
| **LOC** | 161 | 129 | 80.1% |
| **TOTAL** | **977** | **924** | **94.6%** |

*Note: Injected recovery is an automated pipeline smoke test; real-world recall is strictly measured via `gold.py evaluate`.*

---

## 6. Taxonomy Guidelines (Section 9 Compliance)

Annotated and verified under strict human-in-the-loop consensus:
- **All years are DATE**: Standalone years (`2010`, `1818`, `1452`), ranges (`1452-1519`), full dates (`May 9, 1946`), and weekdays (`Sunday`) are labelled `DATE`.
- **All geographic entities are LOC**: Cities (`Florence`, `Milan`, `Mckinney`), countries (`UNITED STATES`, `Italy`), boroughs (`Matanuska-Susitna Borough`), and physiographic regions (`Mat-Su Valley`, `Peloponnese`) are labelled `LOC`.
- **Public & historical figures are NAME**: Historical and public figures (`Leonardo da Vinci`, `Elvis Presley`, `Bernard Madoff`, `Upton Sinclair`, `Candice Bergen`) are labelled `NAME`.
- **Artistic Titles & Creative Media Excluded**: Television series titles and media works (e.g. *"Boston Legal"*, *"Candice Tells All"*) are excluded from `LOC` and `NAME`.
- **Heterogeneous Real-World Identifiers are ID**: User handles (`linda07965`), postal routing zip codes (`75069`), and contract codes (`TAR-1101`) are labelled `ID`.
- **Excluded Categories**: Song titles (`"Good Luck Charm"`), drug brand names (`Sulfacetamide`, `Klaron`, `Ovace`), and corporate/organizational entities (`U.S. Census Bureau`, `Billboard`) are explicitly excluded.
- **No Text Cleaning**: Encoding artifacts and mojibake in MS MARCO are preserved verbatim to guarantee exact character offset mapping.

---

## 7. Dataset Partitions & Separation

- **Dev Set (`data/pii_marco.jsonl`)**: Exactly 1,000 passages from MS MARCO streaming rows 0 to 999.
- **Gold Benchmark (`data/gold_natural.jsonl`)**: Exactly 75 passages from MS MARCO streaming rows 1000+ (50 retained after pruning duplicate drug leaflets + 25 rich contact passages).
- **Partition Independence**: Strictly disjoint ($Row_{Dev} \cap Row_{Gold} = \emptyset$, `SKIP >= 1000`).

---

## 8. License & Acknowledgments

This research is conducted under the mentorship of **Dr. Chandan Biswas** at the **Technology Innovation Hub (IDEAS-TIH), Indian Statistical Institute (ISI), Kolkata**.

Distributed under the MIT License. See `LICENSE` for more information.
