# APCIRA - AI-Powered Cybersecurity Incident Response Assistant

> Semantic retrieval, retrieval-augmented generation (RAG), and machine-learning models over the
> CIC-IDS2017 intrusion-detection dataset, to help security analysts investigate, classify, and
> prioritize network security incidents.

APCIRA turns 2.8 million raw network-flow records into a searchable library of ~97,500 historical
incidents. An analyst can ask a question in plain English ("What historical incidents resemble
DDoS traffic with high packet volume on port 80?"). The system finds the most similar historical
incidents with hybrid keyword and semantic search, and Claude writes an evidence-grounded
assessment with investigation and response recommendations that cite incident IDs.

Separately, three trained models classify the attack type, score intrusion risk, and flag
anomalous traffic from a flow's 24 numeric features.

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Usage](#usage)
- [Configuration](#configuration)
- [Machine-learning models](#machine-learning-models)
- [Project structure](#project-structure)
- [Data and outputs](#data-and-outputs)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## Quick start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested with 3.10.11 |
| Java 17+ | Required by PySpark (data preparation only). Tested with Java 22 |
| ~3 GB free disk | Dependencies (torch, pyspark) plus the ~620 MB Chroma index |
| CIC-IDS2017 `MachineLearningCVE` CSVs | The 8 CSV files go in `MachineLearningCVE/` |
| Anthropic API key | Needed only for the RAG answer step |

### 1. Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python`.

Optional extras: `truststore` (use the OS certificate store behind TLS-inspecting proxies) and
`openai` (only if you point the LLM client at an OpenAI-compatible endpoint).

### 2. Add your API key

Create a `.env` file in the project root. It is gitignored:

```
ANTHROPIC_API_KEY=your-key-here
```

### 3. Build the data and indexes (one-time)

```bash
.venv/Scripts/python scripts/data/prepare_incident_data.py
```

```bash
.venv/Scripts/python scripts/indexing/build_search_indexes.py --rebuild
```

| Step | Produces | Typical time (16-core CPU, no GPU) |
|---|---|---|
| Data preparation | `data/processed/incidents_100k.parquet` | ~3 minutes |
| Index build | `data/processed/bm25_index.pkl`, `data/processed/chroma/` | ~35 minutes (almost all of it embedding 97k texts) |

The first index build downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) from Hugging Face.

### 4. Ask questions

```bash
.venv/Scripts/python scripts/search/interactive_rag.py
```

Startup takes about 15–20 seconds while the data, the indexes and the embedding model load. Then
type a question at the `>` prompt and enter `quit` to exit.

---

## How it works

```
 MachineLearningCVE/*.csv  (8 files, 2,830,743 flows)
            │
            ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 1. DATA PREPARATION  (PySpark)  prepare_incident_data.py │
 │    load → normalize headers → cast numerics → inf→null   │
 │    → standardize labels → hash-dedup → drop null rows    │
 │    → stratified sample → assign INC-IDs → incident_text  │
 └──────────────────────────────────────────────────────────┘
            │  incidents_100k.parquet (97,549 rows × 41 cols, 15 labels)
            ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 2. INDEXING  build_search_indexes.py                     │
 │    BM25 keyword index  +  all-MiniLM-L6-v2 embeddings    │
 │    stored in Chroma (cosine) with incident metadata      │
 └──────────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 3. RAG QUERY  interactive_rag.py → rag/rag_pipeline.py   │
 │    a. infer filters from the question (label, port, ...) │
 │    b. metadata hard filter  → eligible incident IDs      │
 │    c. numeric hard filter   (packet rate, duration, ...) │
 │    d. hybrid search: BM25 + vector, fused with RRF       │
 │    e. re-validate results against every filter           │
 │    f. build context + grounded prompt → Claude           │
 └──────────────────────────────────────────────────────────┘
            │
            ▼
 Structured answer: Assessment · Evidence · Relevance ·
 Investigation · Response · Confidence/Limitations
```

### Incident text

Each sampled flow becomes a template-built description, with no LLM involved. This text is what
gets indexed:

```
Attack: DoS Hulk. Destination port: 80. Flow duration: 101175690 microseconds.
Forward packets: 5. Backward packets: 7. Forward bytes: 379. Backward bytes: 11595. ...
```

Missing values are omitted rather than filled in.

### Hybrid search

- **BM25** ranks exact keyword matches. A custom tokenizer folds case, splits camel case and
  strips common `-ing`/`-ed`/plural suffixes.
- **Vector search** embeds the query with `all-MiniLM-L6-v2` (384 dimensions) and finds the
  nearest incident descriptions in Chroma by cosine distance.
- **Reciprocal Rank Fusion** (k = 60, equal weights) merges the two ranked lists by rank, so the
  differently scaled BM25 scores and cosine distances never have to be compared directly.
- Results with identical text are collapsed so the evidence is not repetitive.

### Hard filters in RAG

Before retrieval, the RAG pipeline reads constraints out of the question and applies them
**strictly**. If nothing matches, it says so rather than falling back to unfiltered results.

| Query phrase | Becomes |
|---|---|
| an attack name, e.g. "PortScan", "DDoS" | `label` filter |
| "port 80", "destination port 443", "src port 22" | `dst_port` / `src_port` filter |
| "TCP", "UDP", "protocol 6" | `protocol` filter (no values in this dataset, see limitations) |
| "high packet volume" / "high traffic volume" | packet rate ≥ the 75th percentile within the filtered set |
| "more than 1000 packets/s", "pps over 500" | explicit packet-rate condition |
| "more than 20 packets" | total packets (forward + backward) condition |
| "duration over 2 seconds" | flow duration, converted to microseconds |

Filters passed with `--filter` override inferred filters on the same field.

---

## Usage

### RAG assistant (`scripts/search/interactive_rag.py`)

```bash
# Interactive: type questions at the prompt
.venv/Scripts/python scripts/search/interactive_rag.py

# Answer one question and exit
.venv/Scripts/python scripts/search/interactive_rag.py "How do PortScan incidents differ from benign traffic?"

# Explicit metadata filters (AND-combined) and fewer evidence incidents
.venv/Scripts/python scripts/search/interactive_rag.py --filter label=PortScan --filter dst_port=80 --top-k 3 "Find similar historical incidents"

# Show filters, numeric thresholds, candidate counts and packet rates of the evidence
.venv/Scripts/python scripts/search/interactive_rag.py --debug "DDoS traffic on port 80 with high packet volume"
```

| Option | Default | Meaning |
|---|---|---|
| `query` (positional) | none | Answer one question, then exit |
| `--top-k N` | 5 | Number of historical incidents passed to the LLM as evidence |
| `--filter FIELD=VALUE` | none | Hard metadata filter; repeatable. Aliases: `attack_type` → `label`, `destination_port` → `dst_port` |
| `--debug` | off | Print the hard-filter trace before the answer |

Example `--debug` trace:

```
Metadata filters: {'label': 'DDoS', 'dst_port': 80}
Numeric: flow_packets_per_sec >= 12.77 packets/s (source=DDoS P75, sample_count=7842)
Candidate counts: {'before_metadata_filter': 97549, 'after_metadata_filter': 7843,
                   'after_numeric_filter': 1961, 'bm25_results': 5, 'vector_results': 5, 'rrf_results': 5}
```

### Hybrid search without an LLM (`scripts/search/interactive_hybrid_search.py`)

Shows the BM25, vector and fused rankings side by side. It needs no API key.

```bash
.venv/Scripts/python scripts/search/interactive_hybrid_search.py "PortScan SYN multiple destinations"
.venv/Scripts/python scripts/search/interactive_hybrid_search.py --debug --filter label=PortScan "PortScan SYN traffic"
```

`--debug` prints the normalized query, the BM25 tokens and each result's RRF contribution. See
[docs/HYBRID_SEARCH_QUERIES.txt](docs/HYBRID_SEARCH_QUERIES.txt) for example queries.

### Retrieval evaluation (`scripts/evaluation/evaluate_hybrid_search.py`)

Runs 12 hand-labelled queries and reports Hit@K, Precision@K and MRR for BM25, vector and RRF.

```bash
.venv/Scripts/python scripts/evaluation/evaluate_hybrid_search.py --top-k 5 --output data/logs/eval_report.txt
```

---

## Configuration

### Environment variables (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Anthropic API key (also accepts `LLM_API_KEY` or `OPENAI_API_KEY`) |
| `LLM_MODEL` | `claude-sonnet-5-5` | Model used for RAG answers (`ANTHROPIC_MODEL` also accepted) |
| `LLM_MAX_TOKENS` | `16000` | Maximum output tokens per answer |
| `LLM_BASE_URL` | none | An OpenAI-compatible or OpenRouter endpoint (requires `openai`) |
| `LLM_TEMPERATURE` | `0` | Used only by the OpenAI-compatible path; current Claude models reject custom sampling values |
| `RAG_CONTEXT_TOP_K` | `5` | Default evidence count when `--top-k` is not given |
| `HF_HUB_OFFLINE` | unset | Set to `1` once the embedding model is cached to skip Hugging Face network checks |

### Code settings (`src/config.py`)

Paths, the raw-to-internal column mapping, the label standardization map, the per-class sampling
targets (`SAMPLES_PER_CLASS_100K`), the random seed (42), the embedding model name, the Chroma
collection name, and the RRF constant and weights.

---

## Machine-learning models

All three models train on `data/cleaned_network_data/cleaned_network_data.csv`, produced by
`ml/eda/data_cleaning.ipynb`. They use the same **24 numeric flow features**, with `dst_port`
deliberately excluded so the models learn traffic behaviour rather than "port X means attack Y".
Duplicate feature vectors are removed before splitting so no flow appears in both train and test.
Each notebook saves `ml/models/<name>.pkl` plus `<name>_meta.json` (features, thresholds, metrics)
and ends with a "test with custom data" cell.

| Notebook | Model | Task | Held-out test results |
|---|---|---|---|
| [`ml/attack_classifier.ipynb`](ml/attack_classifier.ipynb) | Random Forest (balanced classes) | 15-class attack type | Accuracy 0.986 · weighted F1 0.985 · macro F1 0.839 (rare classes such as SQL injection and Infiltration pull the macro score down) |
| [`ml/intrusion_risk.ipynb`](ml/intrusion_risk.ipynb) | Histogram gradient boosting, isotonic-calibrated | Probability a flow is malicious → Low / Medium / High / Critical | ROC-AUC 0.9997 · recall 0.998 · precision 0.991 · false-alarm rate 0.7% |
| [`ml/anomaly_detector.ipynb`](ml/anomaly_detector.ipynb) | arcsinh transform → Isolation Forest, trained on BENIGN only | Unsupervised "how unusual is this flow" → Low / Medium / High | ROC-AUC 0.855 · 5.1% benign false alarms at Medium or above |

The intrusion-risk notebook also runs an **unseen-attack test**. It retrains the model without one
attack family and checks whether that family is still caught:

| Hidden family | Risk model detects | Anomaly detector flags |
|---|---|---|
| Web Attack | 2.8% | 0% |
| Infiltration | 0% | 71% |
| PortScan | 22% | 63% |

The supervised model is very accurate on attacks it has seen, and the anomaly detector adds some
coverage for novel ones. EDA notebooks are in `ml/eda/`.

> These models are not yet wired into the RAG pipeline or exposed through an API.

---

## Project structure

```
Cybersecurity-assistant/
├── MachineLearningCVE/                  Raw CIC-IDS2017 CSVs (8 files, not in git)
├── data/
│   ├── processed/                       Generated, not in git
│   │   ├── incidents_100k.parquet       Prepared incidents
│   │   ├── bm25_index.pkl               BM25 keyword index
│   │   └── chroma/                      Chroma vector store
│   ├── cleaned_network_data/            ML training table (from data_cleaning.ipynb)
│   └── logs/                            Pipeline and index-build logs
├── src/
│   ├── config.py                        Central configuration
│   ├── data/                            PySpark data pipeline
│   │   ├── loader.py                    Load and align the 8 CSVs
│   │   ├── cleaner.py                   Normalize, cast, clean labels, deduplicate
│   │   ├── sampler.py                   Stratified sampling
│   │   ├── text_builder.py              Flow → incident_text
│   │   └── validator.py                 Output validation
│   ├── retrieval/
│   │   ├── bm25_index.py                BM25 index and tokenizer
│   │   ├── embedder.py                  Sentence-transformer wrapper
│   │   ├── chroma_store.py              Chroma build, search and alignment checks
│   │   ├── hybrid.py                    BM25 + vector + RRF
│   │   ├── metadata.py                  Metadata records and frame validation
│   │   ├── metadata_filter.py           Filter validation and eligible IDs
│   │   ├── filters.py                   Chroma-style operator evaluation
│   │   └── numeric_filter.py            Numeric conditions from natural language
│   ├── rag/
│   │   ├── rag_pipeline.py              End-to-end RAG orchestration
│   │   ├── context_builder.py           Evidence → context block
│   │   └── prompt_builder.py            Grounded prompt template
│   └── llm/client.py                    Anthropic / OpenAI-compatible client
├── scripts/
│   ├── data/prepare_incident_data.py
│   ├── indexing/build_search_indexes.py
│   ├── search/interactive_rag.py
│   ├── search/interactive_hybrid_search.py
│   └── evaluation/evaluate_hybrid_search.py
├── ml/
│   ├── attack_classifier.ipynb
│   ├── intrusion_risk.ipynb
│   ├── anomaly_detector.ipynb
│   ├── eda/                             analyse.ipynb, data_cleaning.ipynb
│   └── models/                          Saved models (.pkl) and metadata (.json)
├── docs/                                Additional documentation
├── project_guidelines/                  Capstone brief and submission guidelines
└── requirements.txt
```

---

## Data and outputs

### Sampled dataset (`incidents_100k.parquet`)

97,549 incidents, 41 columns: `incident_id`, `incident_text`, `label`, ports, and flow, packet,
flag, window and timing features. Large classes are sampled down toward fixed targets; small
classes are kept whole.

| Label | Rows | Label | Rows |
|---|---:|---|---:|
| BENIGN | 52,210 | DoS Slowhttptest | 2,584 |
| DDoS | 7,843 | Bot | 1,936 |
| PortScan | 7,828 | Web Attack - Brute Force | 1,323 |
| DoS Hulk | 7,819 | Web Attack - XSS | 651 |
| DoS GoldenEye | 5,246 | Infiltration | 36 |
| FTP-Patator | 4,259 | Web Attack - Sql Injection | 20 |
| SSH-Patator | 3,149 | Heartbleed | 11 |
| DoS slowloris | 2,634 | **Total** | **97,549** |

Of the 2,830,743 raw flows, 2,236,725 remain after hash-based deduplication.

### Generated files

| Path | Size | Rebuild with |
|---|---|---|
| `data/processed/incidents_100k.parquet` | ~20 MB | `prepare_incident_data.py` |
| `data/processed/bm25_index.pkl` | ~130 MB | `build_search_indexes.py --rebuild` |
| `data/processed/chroma/` | ~620 MB | `build_search_indexes.py --rebuild` |
| `data/logs/*.log` | small | written by each run |

If the Parquet changes, rebuild the indexes. The search scripts refuse to run against stale or
misaligned indexes.

---

## Known limitations

- **Metadata filtering is limited to `label` and `dst_port`.** The `MachineLearningCVE` release of
  CIC-IDS2017 has no source/destination IP, source port, protocol or timestamp columns, so those
  fields are empty. Filtering on them returns an error that lists the available fields. The
  "GeneratedLabelledFlows" release of the dataset contains them.
- **Each incident is one flow.** Volumetric or distributed behaviour, such as many sources hitting
  one host, cannot be seen from a single record, and the LLM says so in its limitations section.
- **The answer depends on an external API.** Retrieval works offline once the embedding model is
  cached; the generated answer needs network access and an API key.
- **Not yet implemented** from the capstone brief: REST API endpoints, input guardrails, incident
  prioritization in the RAG flow, model integration into RAG, DeepEval, LLM-as-judge, a multi-agent
  workflow, a feedback loop and a front end.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Retrieval indexes are missing` | Run `build_search_indexes.py --rebuild` |
| `Cached BM25 index does not match the current Parquet data` | The Parquet was regenerated; rebuild the indexes |
| `No LLM API key configured` | Add `ANTHROPIC_API_KEY` to `.env` |
| `ModuleNotFoundError` (pyspark, chromadb, ...) | Use the venv interpreter `.venv/Scripts/python`, not the system Python |
| `pip install` fails building `pyspark` | It is a ~450 MB source build; retry. It is usually a transient download failure |
| Spark `winutils.exe` / `HADOOP_HOME` warning on Windows | Harmless; the pipeline only reads local files |
| Hugging Face "unauthenticated requests" or symlink warnings | Harmless; set `HF_HUB_OFFLINE=1` after the first download to silence network checks |
| `too many SQL variables` from Chroma | Fixed: alignment checks and large candidate sets are now paged or filtered in Python |

---

## Documentation

- [docs/IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md): component-by-component
  implementation details, design decisions and the fix log
- [docs/QUICKSTART.md](docs/QUICKSTART.md): short getting-started guide
- [docs/METADATA_FILTERING.md](docs/METADATA_FILTERING.md): metadata filter fields and behaviour
- [docs/HYBRID_SEARCH_QUERIES.txt](docs/HYBRID_SEARCH_QUERIES.txt): example search queries
- [docs/DELIVERABLES.txt](docs/DELIVERABLES.txt): data-pipeline deliverables checklist
- [project_guidelines/](project_guidelines/): capstone brief (Project 8) and submission guidelines

## Technologies

PySpark · Pandas · PyArrow · rank-bm25 · Sentence Transformers (`all-MiniLM-L6-v2`) · ChromaDB ·
Anthropic Claude · scikit-learn · Jupyter
