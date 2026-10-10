# APCIRA Implementation Summary

How each part of the AI-Powered Cybersecurity Incident Response Assistant is built: what every
component does, the design decisions behind it, how the parts connect, the verified results of
the current build, and the bugs found and fixed along the way.

For setup and day-to-day usage, see the [README](../README.md).

---

## Contents

1. [System overview](#1-system-overview)
2. [Configuration](#2-configuration-srcconfigpy)
3. [Data preparation pipeline](#3-data-preparation-pipeline)
4. [Retrieval layer](#4-retrieval-layer)
5. [RAG layer](#5-rag-layer)
6. [LLM client](#6-llm-client)
7. [Command-line scripts](#7-command-line-scripts)
8. [Machine-learning models](#8-machine-learning-models)
9. [Explainability (SHAP)](#9-explainability-shap)
10. [ML service layer](#10-ml-service-layer)
11. [Incident prioritization](#11-incident-prioritization)
12. [Agent tools](#12-agent-tools)
13. [Multi-agent assistant (LangGraph)](#13-multi-agent-assistant-langgraph)
14. [Guardrails](#14-guardrails)
15. [Agent trace](#15-agent-trace)
16. [LLM providers for the agents](#16-llm-providers-for-the-agents)
17. [LLM-as-Judge](#17-llm-as-judge)
18. [Feedback loop](#18-feedback-loop)
19. [FastAPI backend](#19-fastapi-backend)
20. [Streamlit front end](#20-streamlit-front-end)
21. [Automated tests](#21-automated-tests)
22. [Verified build results](#22-verified-build-results)
23. [Fix log](#23-fix-log)
24. [Design decisions](#24-design-decisions)
25. [Capstone requirement status](#25-capstone-requirement-status)

---

## 1. System overview

```
 ┌────────────────────┐   ┌──────────────────────┐   ┌───────────────────────┐
 │ src/data/  (Spark) │──▶│ src/retrieval/       │──▶│ src/rag/ + src/llm/   │
 │ load, clean,       │   │ BM25 + Chroma +      │   │ filters → hybrid      │
 │ sample, describe   │   │ RRF, metadata and    │   │ search → context →    │
 │                    │   │ numeric filters      │   │ prompt → Claude       │
 └────────────────────┘   └──────────────────────┘   └───────────────────────┘
          │                         ▲                           ▲
          ▼                         │                           │
  incidents_100k.parquet ──▶ bm25_index.pkl + chroma/    interactive_rag.py

 ml/eda/data_cleaning.ipynb ──▶ cleaned_network_data.csv ──▶ ml/*.ipynb ──▶ ml/models/*.pkl
```

The data, retrieval and model-training tracks above feed the assistant layers below:

```
                  Streamlit UI (frontend/) / curl / DeepEval (planned)
                                     │ HTTP (JSON, Server-Sent Events)
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ src/api/  FastAPI: /chat, /chat/stream, /ml/*, /search, /incidents, /health    │
 └──────────────┬──────────────────────────────────────────────┬─────────────────┘
                │ questions                                    │ direct calls (no LLM)
                ▼                                              │
 ┌──────────────────────────────────────────────────────────┐  │
 │ src/agents/  LangGraph assistant                         │  │
 │  input guard → supervisor (router) → specialist agents   │  │
 │  → final answer / follow-up → output guard               │  │
 │  agents: Classification · Risk & Priority ·              │  │
 │          Incident Search · Response                      │  │
 └──────────────┬───────────────────────────────────────────┘  │
                │ tool calls                                   │
                ▼                                              ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ src/agents/tools/  9 LangGraph tools (dicts in, dicts out, never raise)        │
 └───────┬──────────────────────────┬──────────────────────────┬─────────────────┘
         ▼                          ▼                          ▼
  src/services/ml_service.py   src/services/prioritization.py  src/rag/ (retrieval only)
  3 models + SHAP              P1–P4 score + overrides         BM25 + Chroma + RRF + filters
         ▲
  src/services/feature_input.py: parse units, validate, derive, fill + flag, follow-up
```

The project has these parts:

- **Retrieval + RAG** (`src/data`, `src/retrieval`, `src/rag`, `src/llm`): turns raw flows into
  searchable incidents and answers analyst questions grounded in retrieved evidence (§3–6).
- **Machine learning** (`ml/`): trains the attack-type, intrusion-risk and anomaly models on a
  cleaned feature table derived from the same Parquet file, and explains them with SHAP (§8–9).
- **Services** (`src/services`): a model-serving layer that accepts partial user input, plus
  incident prioritization (§10–11). It has no dependency on RAG, the LLM or LangGraph.
- **Agents** (`src/agents`): a LangGraph supervisor that routes each question to the agents it
  needs. The agents call the ML, prioritization and retrieval tools; guardrails wrap the graph,
  and every step is recorded in an agent trace (§12–16).
- **API** (`src/api`): a FastAPI backend that exposes the assistant and the tools over HTTP
  (§19).
- **Front end** (`frontend/`): a Streamlit app for analysts that talks only to the API (§20).

The ML service and RAG retrieval are **separate modules**. They are not one fixed pipeline: the
supervisor decides per question which agents run, and each agent decides which tools to call.

All scripts add `src/` to `sys.path` and resolve paths from their own location. Notebooks locate
the project root by walking up from the working directory to the folder that contains both
`requirements.txt` and `src/`. Nothing depends on an absolute path, so the project runs from any
checkout location.

---

## 2. Configuration (`src/config.py`)

| Setting | Value / purpose |
|---|---|
| `DATA_RAW_DIR` | `MachineLearningCVE/` |
| `DATA_PROCESSED_DIR` | `data/processed/` (created on import) |
| `RAW_CSV_FILES` | The 8 CIC-IDS2017 day files |
| `RAW_TO_INTERNAL_COLUMNS` | Raw header → snake_case internal name, including variants such as `Destination Port`/`Dst Port` |
| `ML_FEATURE_COLUMNS`, `CONTEXT_COLUMNS` | Columns cast to numeric during cleaning |
| `RETRIEVAL_TEXT_COLUMNS` | Columns carried into the Parquet for incident text |
| `LABEL_STANDARDIZATION` | Canonical names for the 15 labels |
| `SAMPLES_PER_CLASS_100K` | Per-class sampling targets (`None` = keep every row) |
| `RANDOM_SEED` | 42 (sampling, shuffle, ML splits) |
| `CLEANED_INCIDENTS_FILE` | `data/processed/incidents_100k.parquet` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `CHROMA_COLLECTION_NAME` | `incidents` |
| `RRF_K`, `BM25_WEIGHT`, `VECTOR_WEIGHT` | 60, 1.0, 1.0 |
| `ML_MODELS_DIR` | `ml/models/` |
| `ATTACK_CLASSIFIER_FILE`, `INTRUSION_RISK_FILE`, `ANOMALY_DETECTOR_FILE` | The three model pickles, each with a `*_META_FILE` JSON (feature order, thresholds, metrics) |
| `FEATURE_REFERENCE_FILE` | `feature_reference.json`: display names, units, BENIGN medians and 1,001 BENIGN quantiles per feature |
| `ATTACK_PROFILES_FILE` | `attack_type_profiles.json`: SHAP-based traffic profile of each attack type |
| `ML_MODEL_VERSIONS_DIR`, `ACTIVE_CLASSIFIER_POINTER` | Retrained classifier versions and the pointer to the active one (`ml/models/versions/`, git-ignored) |
| `FEEDBACK_DB_FILE`, `CLEANED_FLOWS_FILE` | Feedback database; training table used for retraining |

Runtime settings come from `.env` (never committed):

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `google`, `nvidia` or `anthropic` for the agents (§16) |
| `LLM_MODEL` | Model ID for that provider |
| `GOOGLE_API_KEY`, `NVIDIA_API_KEY`, `ANTHROPIC_API_KEY` | Provider keys |
| `NVIDIA_BASE_URL` | NIM endpoint (default `https://integrate.api.nvidia.com/v1`) |
| `LLM_REASONING_EFFORT` | Thinking effort for reasoning models (default `low`) |
| `LLM_REQUESTS_PER_MINUTE` | Optional client-side rate limit (e.g. 5 for the Gemini free tier) |
| `JUDGE_PROVIDER`, `JUDGE_MODEL` | Optional separate model for LLM-as-Judge (default: the main LLM) |
| `JUDGE_MODE` | `on` judges every assistant answer by default; `off` (default) only on request |
| `FEEDBACK_DB_PATH` | Optional location of the feedback database (default `data/feedback/feedback.db`) |
| `API_CORS_ORIGINS` | Allowed browser origins for the API (default: Streamlit on port 8501) |

Sampling targets:

| Label | Target | Label | Target |
|---|---:|---|---:|
| BENIGN | 52,000 | DoS Slowhttptest | 2,600 |
| DoS Hulk | 7,800 | Bot | 2,600 |
| PortScan | 7,800 | Web Attack - Brute Force | 1,300 |
| DDoS | 7,800 | Web Attack - XSS | 1,300 |
| DoS GoldenEye | 5,200 | Web Attack - Sql Injection | 520 |
| FTP-Patator | 5,200 | Infiltration | keep all |
| SSH-Patator | 5,200 | Heartbleed | keep all |
| DoS slowloris | 2,600 | | |

---

## 3. Data preparation pipeline

Entry point: `scripts/data/prepare_incident_data.py`. It is an orchestration script; all logic
lives in `src/data/`. Logs go to the console and `data/logs/data_prep_<timestamp>.log`.

### 3.1 `loader.py`: load and align

- `create_spark_session()` sets 4 GB driver and executor memory and 50 shuffle partitions. It
  also sets `PYSPARK_PYTHON`/`PYSPARK_DRIVER_PYTHON` to the running interpreter, so Spark workers
  use the project venv rather than whatever `python` is on PATH.
- `load_csv_file()` reads one CSV with header and schema inference, normalizes the headers, and
  aligns the file to the canonical schema. Missing optional columns are added as typed nulls; a
  missing `label` or `dst_port` is an error.
- `load_all_cic_ids_files()` loads all 8 files, combines them with `unionByName`, and checkpoints
  the result.
- `verify_required_columns()` checks that the required columns exist.

### 3.2 `cleaner.py`: clean

`clean_and_prepare()` runs six steps in order:

1. **Normalize column names.** Each header is reduced to a canonical key (case, spacing and
   punctuation removed) and matched against `RAW_TO_INTERNAL_COLUMNS`.
2. **Cast numerics.** All feature and port columns are cast to `double`.
3. **Infinity → null.** `±inf` in any numeric column (common in `Flow Packets/s`) becomes null.
4. **Standardize labels.** Labels are trimmed and the Web Attack separator is repaired (see
   [fix 1](#23-fix-log)); then `LABEL_STANDARDIZATION` is applied.
5. **Deduplicate.** A SHA-256 hash over 11 flow-identity fields (IPs, ports, protocol, duration,
   packet and byte counts, label; nulls encoded as `__NULL__`) is built, and duplicates are
   dropped by hash. This avoids a full-row comparison over ~90 columns.
6. **Drop incomplete rows** where `label` or `dst_port` is null.

### 3.3 `sampler.py`: stratified sampling

`stratified_sample()` computes a per-label fraction: `target / count`, or 1.0 when the class is
below its target or marked keep-all. It then calls Spark's `sampleBy` with the fixed seed. Labels
not in the target table get fraction 0, which is why the label repair in step 4 matters.
`verify_sample_distribution()` warns when a class deviates from its target by more than 5%.
Bernoulli sampling makes counts approximate, e.g. Brute Force came out at 1,323 against a target
of 1,300.

### 3.4 Pandas stage (orchestration script)

Only the ~97k sampled rows are converted to Pandas. They are shuffled (seed 42), given IDs
`INC-000001` … `INC-097549`, and each gets an `incident_text`.

### 3.5 `text_builder.py`: incident descriptions

`flow_to_text()` builds a deterministic, template-based description:

- `Traffic label: BENIGN.` for benign rows, `Attack: <label>.` otherwise.
- Protocol (6/17/1 shown as TCP/UDP/ICMP), source and destination ports.
- 34 flow measurements in a fixed order, with units where they apply (microseconds, bytes).
- Missing or NaN values are **omitted**, never shown as `0` or `N/A`. IPs and timestamps are kept
  in the Parquet but excluded from the text.

`build_incident_text_column()` (Spark helper) and `build_text_corpus_pandas()` (legacy) wrap the
same function.

### 3.6 `validator.py`: output checks

`validate_final_dataset()` re-reads the saved Parquet with Spark and checks:

- the required columns are present;
- `dst_port` is not entirely null;
- incident IDs are unique;
- the text column has no nulls or empty strings;
- the label distribution;
- duplicates, using the same hash key as the cleaner;
- nulls in the required columns.

It also prints the schema.

---

## 4. Retrieval layer

### 4.1 `metadata.py`: shared metadata contract

- `validate_incident_frame()` is the gate every script passes before indexing or searching. It
  requires `incident_id`, `label`, `incident_text` and `dst_port`; unique, non-empty IDs; present
  labels and text; and numeric ports.
- `build_incident_metadata()` builds one record per row, in row order: `incident_id`, `label`, and
  `dst_port`, `protocol`, `src_port`, `src_ip`, `dst_ip` and `timestamp` when present. NumPy and
  timestamp values are converted to JSON-safe types.

BM25 and Chroma store the **same** metadata list, which is what makes alignment checking possible.

### 4.2 `bm25_index.py`: keyword search

- **`tokenize()`** splits camel case (only when every part has at least 3 characters), folds case,
  turns punctuation into spaces, and strips simple suffixes: `-ing` and `-ed` (undoubling a
  trailing double consonant) and plural `-s` (except `-ss`, `-us`, `-is`). Documents and queries
  share it.
- **`BM25Index`** wraps `rank_bm25.BM25Okapi`. Negative IDF values, from terms present in more than
  half the documents, are clamped to 0. `search()` supports metadata filters, a `candidate_ids`
  allow-list and exact-text deduplication, and breaks ties deterministically by incident ID. The
  index is saved and loaded as a pickle containing the model, IDs, texts and metadata.

### 4.3 `embedder.py`: embeddings

`IncidentEmbedder` wraps `SentenceTransformer('all-MiniLM-L6-v2')`, which produces 384-dimension
vectors, and encodes in batches of 128. `get_embedding_dimension()` works with both
sentence-transformers 6 (the renamed method) and older versions.

### 4.4 `chroma_store.py`: vector store

- **`build_collection()`** deletes and recreates the collection with cosine distance. It stores the
  embedding dimension and the list of metadata fields in the collection metadata, and adds
  documents in batches of 5,000.
- **`search()`**:
  - validates the query dimension;
  - translates filters to a Chroma `where` clause;
  - restricts to `candidate_ids`. Up to 5,000 IDs (`MAX_WHERE_IN_IDS`) go into a `$in` clause;
    larger sets are filtered in Python, because huge `$in` lists exceed SQLite's bound-variable
    limit;
  - starts with `top_k × 5` results and doubles the request until enough distinct-text results
    remain after deduplication and filtering.
- **`validate_alignment()`** confirms that the stored IDs, document texts and metadata match the
  Parquet exactly. It reads the collection in pages of 5,000 (`VALIDATION_PAGE_SIZE`). It runs at
  the end of every index build and every time a search or RAG script starts.

### 4.5 `filters.py`, `metadata_filter.py`: metadata filtering

- `filters.py` provides `to_chroma_where()` (scalar → `$eq`, several fields → `$and`),
  `filter_field_names()`, and `matches_metadata()`, a Python evaluator for
  `$eq $ne $in $nin $gt $gte $lt $lte $and $or` that matches Chroma's semantics so BM25 and Chroma
  filter identically. `FilterBuilder` offers convenience constructors.
- `MetadataFilter` validates a request against the fields **actually populated** in the index. It
  resolves the aliases `attack_type` → `label` and `destination_port` → `dst_port`, rejects
  duplicate fields and unknown operators, and converts values to the stored type (so
  `--filter dst_port=80` becomes the integer 80). `eligible_ids()` returns the matching IDs.

### 4.6 `numeric_filter.py`: numeric conditions from natural language

`resolve_numeric_conditions()` reads three kinds of numeric condition from a question:

| Phrase pattern | Feature | Notes |
|---|---|---|
| `high packet/traffic volume` | `flow_packets_s` | Threshold = 75th percentile over the metadata-filtered candidates (e.g. "DDoS P75") |
| `<op> N packets/s`, `packet rate <op> N`, `pps` | `flow_packets_s` | Operators: more/greater than, above, over, at least, no less than, less than, below, under, at most, `> >= < <=` |
| `<op> N packets` | total packets | `total_fwd_packets + total_bwd_packets` |
| `duration <op> N <unit>` | `flow_duration` | µs, ms, s, minutes and hours are converted to microseconds |

`apply_numeric_filters()` applies all conditions with AND semantics and returns the surviving IDs.

### 4.7 `hybrid.py`: Reciprocal Rank Fusion

`HybridSearcher.search()`:

1. validates the metadata filters and computes eligible IDs;
2. intersects them with any `candidate_ids`;
3. runs BM25 and vector search with the same restrictions;
4. fuses the results with weighted RRF: `score = Σ wᵢ / (k + rankᵢ)`, k = 60.

It returns the `fused`, `bm25` and `vector` lists plus metadata for every returned ID. With
`debug=True` it adds the normalized query, the BM25 tokens and each result's rank contributions.

---

## 5. RAG layer

### 5.1 `rag_pipeline.py`: orchestration

`rag_pipeline(query, filters, top_k, searcher, df, llm_client)`:

1. **Validate** that the query is non-empty and `top_k` ≥ 1.
2. **Load** the searcher if none is supplied (`_default_hybrid_searcher()`): it reads the Parquet,
   checks that the BM25 IDs, texts and metadata equal the Parquet, validates Chroma alignment, and
   loads the embedder. Callers that answer many questions load this once and pass it in.
3. **Infer filters** from the question (`_extract_query_filters()`), matching only values that exist
   in the index:
   - attack labels (longest first, whole-phrase matches);
   - protocol names or numbers;
   - "source/destination port N" phrases;
   - source/destination IPs;
   - ISO timestamps.
4. **Merge** them with explicit filters, which win on the same field.
5. **Apply the metadata hard filter.** If nothing is eligible, return "No matching historical
   incidents were found." without calling the LLM.
6. **Apply numeric hard filters** to the eligible rows. Zero survivors also stops before the LLM.
7. **Run hybrid search** restricted to the surviving candidate IDs.
8. **Build records** for the returned IDs only, joining the full Parquet row with the index
   metadata, ranked by fused score.
9. **Re-validate** every record against the metadata and numeric filters, drop and log any
   violations, and stop if none remain.
10. **Build the context and prompt** and call the LLM. Provider errors are returned as
    `LLM generation failed: …` instead of raising.

It returns a dict with:

- the query, filters, inferred filters and numeric conditions;
- `candidate_counts` (before and after each filter, and per retriever);
- the result records, context, prompt and answer;
- `evidence_available`;
- the raw search results.

`RAGPipeline` is a small wrapper that holds a searcher, data frame and client.

### 5.2 `context_builder.py`

Formats up to `max_results` incidents (default `RAG_CONTEXT_TOP_K`, 5) as numbered
`=== HISTORICAL INCIDENT n ===` blocks. Each block has the incident ID, attack type, destination
port, retrieval score, the traffic description, and an alphabetized list of additional fields.
Missing values show as `N/A`. When numeric conditions were applied, the pipeline puts them above
the context together with the eligible-candidate count.

### 5.3 `prompt_builder.py`

The prompt tells the model that the retrieved incidents are **historical evidence, not the user's
current incident**. It forbids inventing IPs, ports, timestamps or values, requires incident-ID
citations, and asks the model to say explicitly when the evidence is insufficient or conflicting.
The answer format has six sections: Assessment, Relevant Historical Evidence, Why It Is Relevant,
Recommended Investigation, Recommended Response, and Confidence / Limitations.

---

## 6. LLM client (`src/llm/client.py`)

- Loads `.env` from the project root, using `python-dotenv` when installed and a built-in parser
  otherwise.
- **Shared provider setting.** When `LLM_PROVIDER` is set (`nvidia`, `google` or `anthropic`), the
  key, model, endpoint and reasoning effort come from `src/llm/settings.py`, the same settings the
  agents use (§16), so one `.env` switch changes the LLM for the whole project. NVIDIA NIM and Gemini
  are called through their OpenAI-compatible endpoints, and `reasoning_effort` is passed when set.
- Model output is cleaned by `llm/text.py`: non-breaking hyphens and narrow/thin spaces are replaced
  with plain ones, because the Windows console can't print them.
- Provider selection without `LLM_PROVIDER` (original behaviour):

  | Condition | Provider |
  |---|---|
  | base URL contains `openrouter` | OpenRouter |
  | model name starts with `claude` | Anthropic |
  | base URL or `OPENAI_API_KEY` set | OpenAI-compatible |

- Defaults: model `claude-sonnet-5-5`, `max_tokens` 16,000. These come from `LLM_MODEL` and
  `LLM_MAX_TOKENS` when the constructor arguments are omitted.
- **Anthropic path:** sends no sampling parameters, because current Claude models reject
  non-default values. It joins all `text` blocks from the response, since responses can begin with
  `thinking` blocks.
- **OpenAI-compatible path:** sends `temperature` (`LLM_TEMPERATURE`) and a system message.

---

## 7. Command-line scripts

| Script | Purpose | Key behaviour |
|---|---|---|
| `scripts/data/prepare_incident_data.py` | Raw CSVs → Parquet | 7 logged steps; exits non-zero on any failure |
| `scripts/indexing/build_search_indexes.py` | Parquet → BM25 + Chroma | Skips if the indexes exist (use `--rebuild`). BM25 is written to a `.tmp` file and moved into place only after BM25 and Chroma both pass verification |
| `scripts/search/interactive_rag.py` | Question → grounded answer | Loads the searcher, data and LLM client **once**, then loops; `quit`, Ctrl+C or EOF exits; invalid queries print an error and continue |
| `scripts/search/interactive_hybrid_search.py` | Retrieval explorer, no LLM | Shows BM25, vector and RRF lists; stale or misaligned indexes are refused |
| `scripts/evaluation/evaluate_hybrid_search.py` | Retrieval metrics | 12 labelled queries; Hit@K, Precision@K and MRR per method; optional `--output` report |
| `scripts/ml/interactive_ml.py` | Query the ML service directly | Metrics are extracted from a sentence by the LLM, or by a rule-based parser with `--no-llm`; shows the extracted metrics, then attack type, risk, anomaly, SHAP reasons and priority. Commands: `incident <ID>`, `example <type>`, `profile <type>`, `types`, `mode explain\|predict`, `json on\|off`, `reset`. A follow-up answer is merged with the metrics already given |
| `scripts/agents/interactive_assistant.py` | Chat with the multi-agent assistant | Prints the live agent trace (guardrails, route, agents, tools, timings), then the answer. Conversation memory per session; commands `new`, `trace on\|off`, `judge on\|off`, `feedback up\|down\|correct\|stats`, `quit`; `--no-trace`, `--judge`; a question as argument answers once and exits |
| `scripts/evaluation/judge_answers.py` | LLM-as-Judge evaluation | `--calibrate` checks the judge on answers with known flaws; default runs a 15-question set with the judge on (`--limit`, `--route`) and writes a JSONL report |
| `scripts/feedback/feedback_report.py` | Feedback summary | Approval, judge vs analyst agreement, corrections, comments; `--list`, `--json` |
| `scripts/feedback/export_corrections.py` | Corrections CSV | 24 features + corrected label for retraining |
| `scripts/feedback/retrain_classifier.py` | Retrain the classifier with corrections | `--dry-run`; saves a version and activates it only if test macro F1 does not drop; `--list`, `--activate` |
| `scripts/api/run_api.py` | Start the FastAPI backend | Uvicorn on `127.0.0.1:8000` by default; `--port`, `--host`, `--reload`, `--log-level`; Swagger UI at `/docs` |
| `scripts/frontend/run_frontend.py` | Start the Streamlit front end | `localhost:8501`; `--port`, `--address`, `--api` (§20) |

The indexing and evaluation scripts disable Chroma telemetry by default. They no longer force
Hugging Face offline mode (see [fix 7](#23-fix-log)); set `HF_HUB_OFFLINE=1` yourself once the
model is cached.

---

## 8. Machine-learning models

### 8.1 Training data (`ml/eda/data_cleaning.ipynb`)

Starts from `incidents_100k.parquet` and:

- drops the all-null columns (IPs, source port, protocol, timestamp);
- drops the constant columns;
- drops redundant flag columns;
- drops rows with negative flow duration;
- fills the few missing `flow_packets_s` values with the median;
- removes `incident_id`/`incident_text`.

The result is `data/cleaned_network_data/cleaned_network_data.csv`: label, `dst_port` and 24
numeric features, 97,548 rows with all 15 labels. `ml/eda/analyse.ipynb` is a Spark exploration
of the raw DDoS file (nulls, infinities, constant columns, per-label medians, correlations).

### 8.2 Shared conventions

- The 24 features exclude `dst_port`, so the models learn behaviour rather than port shortcuts.
- Duplicate feature vectors are removed **before** splitting, to prevent train/test leakage.
- The seed is 42, and CPU parallelism is capped at 2 jobs.
- Each notebook saves `ml/models/<name>.pkl` (joblib) and `<name>_meta.json` (feature order,
  thresholds, metrics, scikit-learn version), runs a load-and-predict check, and ends with an
  editable "test with custom data" cell.

### 8.3 Attack-type classifier (`ml/attack_classifier.ipynb`)

Random Forest, 100 trees, `class_weight="balanced"`, 80/20 stratified split, 5-fold stratified
cross-validation on the training set.

| Metric | Value |
|---|---|
| Cross-validated macro F1 | 0.867 ± 0.024 |
| Test accuracy | 0.986 |
| Test weighted F1 | 0.985 |
| Test macro F1 | 0.839 |

Large classes score ≥ 0.98 F1. Macro F1 is pulled down by the tiny classes (SQL injection with 20
rows, Infiltration 36, Heartbleed 11) and by the overlap between Web Attack Brute Force and XSS.

### 8.4 Intrusion-risk model (`ml/intrusion_risk.ipynb`)

- Binary target `is_intrusion`, with a 60/20/20 train/calibration/test split stratified on the
  original label.
- Random Forest and HistGradientBoosting were compared with 5-fold cross-validation;
  HistGradientBoosting was selected on PR-AUC (0.99965 vs 0.99709).
- Probabilities are **isotonic-calibrated** on the calibration set (Brier score 0.0038).
- The decision threshold of 0.406 maximizes F2 (recall-weighted) on the calibration set.
- Risk bands are Critical ≥ 0.9, High ≥ 0.7, Medium ≥ 0.3, otherwise Low.

| Test metric | Value |
|---|---|
| ROC-AUC / PR-AUC | 0.9997 / 0.9996 |
| Recall / precision / F1 | 0.998 / 0.991 / 0.994 |
| False-alarm rate | 0.69% (69 of 9,967 benign) |
| Missed attacks | 16 of 7,478 |

Permutation feature importance is computed on a 5,000-row test sample.

### 8.5 Anomaly detector (`ml/anomaly_detector.ipynb`)

- Pipeline: `arcsinh` transform → Isolation Forest (200 trees), trained on **BENIGN only**.
- BENIGN is split 60/20/20; every attack row is used only for testing.
- The `arcsinh` transform raised ROC-AUC from 0.806 (raw features) to 0.855.
- Score = `-score_samples(X)`, so higher means more anomalous. Thresholds come from the BENIGN
  validation set: Medium ≥ 95th percentile (0.578), High ≥ 99th percentile (0.615).

| Test metric | Value |
|---|---|
| ROC-AUC / average precision | 0.855 / 0.934 |
| Benign false alarms (Medium+ / High) | 5.1% / 1.0% |
| Attacks flagged (Medium+ / High) | 25.6% / 3.9% |

Detection is strong for unusual families (Heartbleed 100%, Infiltration 81%, DoS Hulk 71%,
Slowhttptest 64%, PortScan 60%). It is near zero for attacks that look like normal web or login
traffic (Web Attacks, FTP/SSH-Patator, GoldenEye).

### 8.6 Unseen-attack test (in `intrusion_risk.ipynb`)

The risk model is retrained with one attack family hidden from training:

| Hidden family | Test rows | Risk model detects | Anomaly detector flags (Medium+) | Full model detects |
|---|---:|---:|---:|---:|
| Web Attack | 398 | 2.8% | 0% | 98.5% |
| Infiltration | 7 | 0% | 71% | 71% |
| PortScan | 78 | 22% | 63% | 100% |

Conclusion: the supervised model is excellent on known attack families but weak on unseen ones.
The anomaly detector adds partial coverage for novel attacks.

### 8.7 Housekeeping

`.gitignore` excludes `*.json`. The exception `!ml/models/*.json` now versions the model metadata,
the feature reference and the attack profiles alongside the `.pkl` files. The `tests/` folder was
also ignored; that line was removed so the test suite is tracked.

Notebooks run on the project venv through a registered Jupyter kernel (`cyber-venv`, installed
with `python -m ipykernel install --sys-prefix`), so the saved models match the venv's
scikit-learn version (1.7.2). To execute a notebook headless:
`.venv/Scripts/python -m nbconvert --execute --inplace <notebook>`.

---

## 9. Explainability (SHAP)

Notebook: `ml/explainability.ipynb`. It explains the attack-type classifier and the intrusion-risk
model with SHAP, both globally (which features matter for each attack type) and per prediction
(why this flow got this result).

### 9.1 Explainers

| Model | Explainer | Notes |
|---|---|---|
| Random Forest classifier | `shap.TreeExplainer` | One SHAP vector per class; about 83 ms per flow |
| Calibrated risk model | `shap.TreeExplainer` on the HistGradientBoosting base model | The calibrated wrapper is unwrapped (`calibrated_classifiers_[0].estimator.estimator`); SHAP values are in the base model's log-odds space, and the reported probability stays the calibrated one |

Checks run in the notebook:

- **Additivity:** base value + Σ SHAP equals the model output, with an error of about 1e-13.
- **Feature order:** all three models share the same 24-feature order (asserted).
- **Speed:** fast enough to explain single flows on request; the services compute SHAP lazily,
  only when an explanation is asked for.

### 9.2 Outputs

| Output | Contents |
|---|---|
| `ml/models/attack_type_profiles.json` | For each of the 14 attack types: the top features by share of mean \|SHAP\| (from 150 explained rows per type), the attack median vs the BENIGN median, the median's BENIGN percentile, and whether the feature is higher or lower than normal. Used by the `get_attack_profile` tool |
| `ml/models/feature_reference.json` | For each of the 24 features: display name, unit, BENIGN median, and BENIGN quantiles at 1,001 levels, used to say "higher than 97% of normal traffic" |
| `ml/reports/figures/` (9 PNGs) | Attack-type × feature SHAP heatmap; beeswarm plots for Bot, DDoS, PortScan and Web Attack Brute Force; risk-model beeswarm; `init_win_bytes_forward` dependence plot; waterfall examples for both models |

### 9.3 Findings

- Each attack type has a distinct signature, e.g. DDoS: large backward bytes and backward packet
  sizes; PortScan: tiny flows with almost no payload.
- **Dataset artifact:** `init_win_bytes_forward` (the initial TCP window) accounts for **28.9%** of
  the risk model's total \|SHAP\|. In CIC-IDS2017 the attack tools often use distinctive window
  sizes, so the model partly learns the tooling rather than the behaviour. This is documented as
  a limitation; the value is shown to users like any other reason, not hidden.
- Percentages are capped at 99.9% in wording ("higher than 99.9% of normal traffic"), never 100%.

---

## 10. ML service layer

`src/services/` serves the three models to every caller (scripts, agent tools, API). It has no
dependency on RAG, the LLM or LangGraph.

### 10.1 `feature_input.py`: partial input → 24 features ("fill + flag")

Users rarely know all 24 flow metrics. `normalize_flow(data)` turns any subset into a complete,
validated feature vector and records exactly what was given, derived and filled:

1. **Parse.** Numbers or strings with units: durations (`46 us`, `2 ms`, `1.5 s`, `2 min`, `1 h` →
   microseconds), bytes (`1.2 KB`, `3 MB`), counts and rates (`43k`, `1.2M packets/s`). Friendly
   aliases are accepted (`duration`, `fwd_packets`, `forward_bytes`, `packets_per_second`, raw CSV
   headers such as `Flow Duration`, …).
2. **Validate** (an input guardrail):
   - finite numbers only; values ≥ 0 (initial windows ≥ -1, where -1 means "not set");
   - TCP flags are 0/1 (larger values are treated as 1, with a warning);
   - **physical plausibility** (rejected with a clear message): a flow longer than 7 days, a packet
     size over 65,535 bytes, or more bytes than `packets × 65,535` in a direction;
   - consistency warnings (e.g. a stated packet rate that disagrees with packets ÷ duration by more
     than 50%, or min packet size above max).
3. **Derive** what follows from the given values, using formulas checked against the dataset:
   packet rates per direction and overall (packets ÷ duration), mean packet size per direction
   (bytes ÷ packets), `down_up_ratio = floor(bwd ÷ fwd)`, and `packet_length_mean ≈ bytes ÷
   (packets + 1)` (CIC-IDS2017's own convention; flagged as approximate).
4. **Check key features.** Three groups are required: flow duration; packet counts (forward or
   backward); byte counts (forward or backward). If any group is missing, **no model runs** and the
   result is `needs_follow_up` with a plain-language question (`follow_up_question()`).
5. **Fill** the rest with BENIGN medians and list them in `filled`.
6. **Rate input quality** by the number of provided + derived features: **full** ≥ 20, **reduced**
   ≥ 12, otherwise **low**.

`NormalizedFlow` carries `features`, `provided`, `derived`, `filled`, `missing_key_features`,
`needs_follow_up`, `input_quality` and `warnings`. `get_incident_features(id)` loads a historical
incident's real features (lenient mode: invalid stored values become "missing" instead of
errors), with its historical label and destination port; IDs are normalized (`inc-35` →
`INC-000035`).

### 10.2 `ml_service.py`: the models behind one interface

`get_ml_service()` returns a thread-safe singleton that loads the three models and the reference
files once.

| Method | Returns |
|---|---|
| `classify(flow)` | Attack type, confidence, top 3 alternatives |
| `predict_risk(flow)` | Calibrated intrusion probability, `is_intrusion` (threshold 0.406), risk level |
| `detect_anomaly(flow)` | Anomaly score, its BENIGN percentile, level (Low/Medium/High), and the most unusual metrics (outside the 5th–95th BENIGN percentile) |
| `predict(flow)` | All three, no SHAP (fast; used by prioritization and batches) |
| `predict_batch(flows)` | `predict()` for many flows, vectorized |
| `explain(flow, top_k=3)` | All three + the top SHAP reasons for the attack type and for the risk + a plain-language summary |
| `attack_profile(type)`, `list_attack_types()` | Data from `attack_type_profiles.json` |

Every result includes an `input` summary (quality, provided, derived, filled, warnings) and an
`overall_confidence`: **low** if input quality is low or model confidence < 0.5, **medium** if
input quality is reduced or confidence < 0.8, otherwise **high**.

Explanations:

- Filled features are **excluded from the reasons**: a value the user never gave can't be a
  reason.
- Each reason states the value, how it compares with normal traffic ("higher than 94% of normal
  traffic"), and whether it pushes towards or away from the prediction.
- The summary's wording depends on input quality: low → "indicative only", reduced →
  "reduced-confidence", full → no caveat beyond naming any filled values.

---

## 11. Incident prioritization

`src/services/prioritization.py` turns the three model outputs into a priority.

```
score = 100 × ( 0.45 × risk probability
              + 0.35 × attack-type severity × classifier confidence
              + 0.20 × anomaly component )

anomaly component = clip((BENIGN percentile − 90) / 10, 0, 1)
```

| Priority | Label | Score | Action |
|---|---|---|---|
| P1 | Critical | ≥ 75 | Investigate now |
| P2 | High | ≥ 50 | Investigate today |
| P3 | Medium | ≥ 25 | Review when possible |
| P4 | Low | < 25 | Monitor |

Severity weights: Critical 1.0 (Heartbleed, Infiltration, SQL injection), High 0.8 (DDoS, DoS
variants, Bot, XSS, Web brute force), Medium 0.6 (SSH/FTP-Patator), Low 0.4 (PortScan), BENIGN 0.
Unknown types default to High.

Overrides after scoring:

1. Classified BENIGN but anomaly **High** → at least **P2** (possible unknown attack).
2. Critical-severity type with risk ≥ 0.9 → **P1**.
3. Low-severity reconnaissance (PortScan) is capped at **P3** unless anomaly is High.
4. Low input quality keeps the priority but adds an "indicative only" note.

Each result has the priority, score, label, action, a reason, the three components and the
overrides applied. `prioritize_batch()` ranks many flows by priority, then score; flows that need a
follow-up go last.

The weights were checked on a stratified sample (200 flows per type). No weight setting put
PortScan in P3 without also demoting brute-force and web attacks, hence override 3. After the
override, 97% of PortScan flows are P3 and 97% of BENIGN flows are P4.

---

## 12. Agent tools

`src/agents/tools/` wraps the services as LangGraph `StructuredTool`s. Every tool takes validated
Pydantic arguments, returns a compact JSON-safe dict, and **never raises**: bad input becomes
`{"status": "error", "error": "..."}` (`safe_tool`), so the agent can correct the call or explain
the problem. Floats are rounded, and long input summaries are shortened for the LLM.

| Tool | Wraps | Input |
|---|---|---|
| `classify_attack_type` | `classify()` | Any subset of the 24 metrics (numbers or unit strings) **or** an `incident_id` |
| `assess_intrusion_risk` | `predict_risk()` + `detect_anomaly()` | same |
| `explain_flow` | `explain()` + `prioritize()` | same |
| `prioritize_flow` | `predict()` + `prioritize()` | same |
| `prioritize_flows` | `prioritize_batch()` | Up to 1,000 flows and/or incident IDs, `top_n` |
| `get_attack_profile` | `attack_profile()` + severity | Attack type (case-insensitive, unique substring allowed) |
| `list_attack_types` | `list_attack_types()` | – |
| `lookup_incident` | `get_incident_features()` | Incident ID |
| `search_similar_incidents` | `rag_pipeline()` **retrieval only** | Query, optional `attack_type`, `dst_port`, `top_k` ≤ 20 |

When an `incident_id` is given, results include the incident's `historical_label` and port, kept
separate from the model's prediction.

**Retrieval only.** `search_similar_incidents` runs the full RAG retrieval (inferred and explicit
metadata filters, numeric conditions, hybrid search) but passes a `_RetrievalOnlyClient` whose
`generate()` returns an empty string, so no answer is generated inside the tool. Writing the answer
is the job of the agents and the final-answer node. The searcher loads once (lazy, thread-safe
singleton). The tool returns the filters applied, filters inferred from the query, numeric
conditions, candidate count and, per incident, the ID, attack type, port, relevance score and the
first 10 sentences of its description.

Tools per agent (`TOOLS_BY_AGENT`):

| Agent | Tools |
|---|---|
| Classification | `classify_attack_type`, `explain_flow`, `get_attack_profile`, `list_attack_types`, `lookup_incident` |
| Risk & Priority | `assess_intrusion_risk`, `prioritize_flow`, `prioritize_flows`, `lookup_incident` |
| Incident Search | `search_similar_incidents`, `get_attack_profile`, `lookup_incident` |
| Response | `search_similar_incidents`, `get_attack_profile` |

---

## 13. Multi-agent assistant (LangGraph)

### 13.1 Graph

```
START → input_guard ─(blocked)──────────────────────────────────────────→ END
            │
            ▼
        supervisor ──→ [agents in the route's order] ──→ final_answer ──→ output_guard → END
                                   └──(key metrics missing)──→ ask_follow_up ──┘
```

Built in `src/agents/graph.py` with `StateGraph(AssistantState)` and an `InMemorySaver`
checkpointer, so each `thread_id` is a conversation with memory. Conditional edges after the
supervisor and after every agent call `next_step()`: `ask_follow_up` if a follow-up is pending,
else the first agent in the plan that hasn't run yet, else `final_answer`.

### 13.2 Supervisor (router)

The supervisor asks the LLM for a `RouteDecision` (route + one-sentence reason): first as a plain
request with JSON instructions, parsed and validated with Pydantic; if that reply is not a valid
decision, with `with_structured_output(..., method="json_schema", include_raw=True)`. (Guided
`json_schema` decoding alone sometimes ran away to the token limit on gpt-oss, see fix #33.)

| Route | Agents run | Example |
|---|---|---|
| `classify` | Classification | "What attack is INC-004393?" |
| `risk` | Risk & Priority | "Is this flow suspicious?", "Which of these is most urgent?" |
| `search` | Incident Search | "Has an SSH brute force happened before?" |
| `recommend` | Response | "How should I respond to a DoS Hulk attack?" |
| `investigate` | Classification → Risk & Priority → Incident Search → Response | "Investigate INC-000035 and tell me how to respond" |
| `general` | none (final answer only) | "What is a DDoS?", "What can you do?" |
| `out_of_scope` | none (fixed reply) | Unrelated requests, malware or attack requests |

`route_message()` never raises: a model refusal becomes `out_of_scope`; an unparsable answer or a
failed call falls back to `general`. When the previous turn asked for missing metrics, the router
is told so and keeps the same route for the answer. The supervisor also resets the per-turn state
and keeps the last analysed flow (`previous_flow_args`) for references like "what about its risk?".

### 13.3 Specialist agents

Each agent (`src/agents/specialists.py`) is the LLM bound to its tool set (`bind_tools`) with its
own system prompt (`src/agents/prompts.py`), running a tool-calling loop:

- up to **6 tool rounds**; when the budget is used up, the agent is told to stop calling tools and
  give its findings;
- the task message contains the last 8 conversation messages, the flow already identified this turn
  (or earlier in the conversation), the latest message, and the findings of earlier agents this
  turn;
- if a tool returns `needs_follow_up`, or the agent replies with a line starting `QUESTION:`, the
  turn ends with that question to the user;
- flow arguments from successful flow tools are shared with later agents, so they analyse the
  same flow without asking again.

Shared rules in every agent prompt: use only tool results for facts; never invent metric values,
IDs, ports, IPs, timestamps or attack types; pass metrics exactly as the user gave them; keep the
historical label and the model prediction distinct; always state input-quality caveats. The
Incident Search agent sets the attack-type/port filters when known and follows the RAG evidence
rules. The Response agent gives investigation, containment and remediation steps and marks
general best practice as such.

### 13.4 Final answer

The `final_answer` node writes the single reply from the agents' findings and the conversation,
using only the findings: no new facts, numbers, IDs or attack types; confidence caveats always
stated; historical incident IDs cited. Sections, when relevant: Assessment, Why, Similar past
incidents, Recommended actions, Confidence & limitations. `general` questions are answered briefly
from general knowledge; `out_of_scope` gets a fixed reply without an LLM call. An empty model
reply falls back to the raw findings.

### 13.5 Using it from Python

```python
from agents.graph import Assistant

assistant = Assistant()                       # default LLM from .env
thread = assistant.new_thread_id()
result = assistant.ask("What attack is INC-004393?", thread)
for kind, payload in assistant.stream("What about its risk?", thread):
    ...                                       # ("event", trace event) ..., then ("result", result)
```

The result has `thread_id`, `answer`, `route`, `route_reason`, `agents_run`, `needs_follow_up`,
`findings`, `evidence` (every tool call with its arguments and result), `flow_args`, `trace` and
`guardrails`.

---

## 14. Guardrails

`src/agents/guardrails.py` holds rule-based checks: no LLM calls, deterministic and fast. They
run as the first and last graph nodes.

### 14.1 Input guard (before the supervisor)

| Check | Action |
|---|---|
| Message over 4,000 characters | Trimmed, noted in the trace |
| **Prompt injection**: "ignore previous instructions", "what is your system prompt", "you are now DAN", jailbreak/developer mode, fake `SYSTEM:` lines or `<\|im_start\|>` tags, "new instructions:"; text is NFKC-normalized and zero-width characters are removed first | **Blocked** with a fixed reply |
| **Harmful requests**: writing malware (ransomware, keylogger, reverse shell, …), "how do I launch a DDoS against …", hacking into systems, bypassing antivirus/EDR/firewalls, stealing credentials | **Blocked** with a fixed reply |
| **Secrets and personal data**: API keys (Anthropic/OpenAI, AWS, GitHub, Slack, Google), bearer tokens, `password is …`, `api_key=…`, private keys, emails, card numbers (Luhn-checked) | **Redacted** to `[REDACTED_<KIND>]` before any LLM sees the message |

Defensive questions are allowed: harmful patterns are skipped when the message is about
detecting, preventing, responding to or explaining an attack ("how do I detect a keylogger",
"write a detection rule for ransomware", "how do attackers evade IDS"). A blocked message is
replaced in the conversation history by `[message blocked by input guardrail: <category>]`, so
it can't influence later turns. Blocked messages cost no LLM call.

### 14.2 Tool level

Flow plausibility checks in `feature_input.py` (§10.1) reject impossible values; the agent then
asks the user to correct them instead of guessing.

### 14.3 Output guard (after the final answer or follow-up question)

| Check | Action |
|---|---|
| Incident IDs not found in this turn's tool results or the conversation | Marked `(unverified ID)` |
| Percentages not within 1 point of any number in the tool results (only when model results exist) | Flagged in a note |
| A P1–P4 priority that contradicts the prioritization tool | Flagged in a note |
| Dangerous commands: `rm -rf /`, disk formatting/overwriting, fork bombs, `chmod 777 /`, flushing or disabling the firewall, disabling antivirus/SELinux, `curl … \| sh` | The line is replaced by `[command removed by safety guardrail: <reason>]` |
| Partial input (low/reduced quality) or low confidence, not already stated | Confidence note added |
| Model prediction differs from an incident's historical label, not mentioned | Note added |
| Secrets in the answer | Redacted |
| Answer repeats system-prompt text (fragments of 60+ characters) | Whole reply replaced |

The output guard fixes or annotates the answer; it doesn't ask the LLM to rewrite it. The guarded
text is what is stored in the conversation history.

Limits: pattern rules can miss cleverly reworded attacks. The router's `out_of_scope` route is the
second layer; LLM-as-Judge (§17) is the third.

---

## 15. Agent trace

`src/agents/trace.py` records every step of a turn. `emit()` appends the event to the turn's
trace (returned with every answer) and also sends it live through LangGraph's custom stream
(`get_stream_writer`), so callers can show progress while the graph runs.

| Event | Fields |
|---|---|
| `guard_input` | action (allowed / redacted / blocked), category, rule matched, redactions, truncated |
| `route` | route, reason, plan, elapsed_ms |
| `agent_start` | agent |
| `tool_call` | agent, tool, args |
| `tool_result` | agent, tool, args, status, one-line summary, elapsed_ms |
| `agent_end` | agent, outcome (findings / follow_up), LLM calls, tool calls, elapsed_ms |
| `follow_up` | question |
| `final` | number of findings used, elapsed_ms |
| `guard_output` | issues found and fixed (empty = passed) |

`summarize_tool_result()` writes one line per tool, e.g.
`INC-000035: SSH-Patator (100%), risk Critical, anomaly Low, P2`. `format_event()` renders the
console view used by the chat script:

```
[GUARD-IN]  allowed
[SUPERVISOR] route = investigate  (2.8s)
             plan:   Classification agent -> Risk & Priority agent -> Incident Search agent -> Response agent
[AGENT] Classification agent
    [TOOL] explain_flow(incident_id=INC-000035)
           -> INC-000035: SSH-Patator (100%), risk Critical, anomaly Low, P2  (2.4s)
    done: findings ready; 2 LLM call(s), 2 tool call(s)  (10.1s)
...
[FINAL ANSWER] written from 4 agent finding(s)  (7.6s)
[GUARD-OUT] passed, no changes
```

---

## 16. LLM providers for the agents

Provider settings live in `src/llm/settings.py` and are shared by the agents and by `LLMClient`
(standalone RAG answers, metric extraction in `interactive_ml.py`). **Current setting: NVIDIA NIM
with `openai/gpt-oss-20b`** for the whole project. To switch (e.g. to a new Anthropic key), change
only `LLM_PROVIDER` and `LLM_MODEL` in `.env`.

`src/agents/llm.py` builds the agents' chat model (`get_chat_model()`, cached):

| `LLM_PROVIDER` | Client | Default model | Notes |
|---|---|---|---|
| `google` | `ChatGoogleGenerativeAI` (native Gemini client) | `gemini-3.5-flash` | Keeps Gemini's thought signatures across tool-call turns; `reasoning_effort` from `LLM_REASONING_EFFORT` |
| `nvidia` | `ChatOpenAI` against the NIM OpenAI-compatible endpoint | `openai/gpt-oss-20b` | Longer timeout for free-tier queueing; ~40 requests/minute on the free tier |
| `anthropic` | `ChatAnthropic` | `claude-sonnet-5-5` | No sampling parameters (current Claude models reject non-default values) |

- Without `LLM_PROVIDER`, the provider is inferred from which key is set.
- `LLM_REQUESTS_PER_MINUTE` adds a client-side rate limiter (`InMemoryRateLimiter`, bucket size
  1), so calls are spaced evenly and wait instead of failing with 429.
- `llm_settings()` reports the provider, model and effort without the key; the API's `/health`
  shows `provider:model`.
- The model must support tool calling and JSON output.

Provider evaluation (2026-10-10), measured with a tool-calling request and a JSON route decision:

| Model | Tool call | Route JSON | Time per call | Outcome |
|---|---|---|---|---|
| Anthropic `claude-sonnet-5-5` | ✅ | ✅ | 2–12 s | Used for development; account later ran out of credit |
| NVIDIA `moonshotai/kimi-k3` | ✅ | – | ~125 s (free-tier queue) | Works but too slow (~20 min per investigation) |
| NVIDIA `openai/gpt-oss-20b` | ✅ | ✅ | 2–4 s | **Current setting**: fast, ~40 requests/minute; weaker than Claude at keeping to the evidence (adds general-knowledge claims) |
| NVIDIA `meta/llama-3.3-70b-instruct`, `moonshotai/kimi-k2-instruct` | – | – | – | Retired on NIM (HTTP 410) |
| Google `gemini-3.5-flash` | ✅ | ✅ | ~3 s | Good, but the free tier allows only 20 requests/day |
| Google `gemini-3.8-flash` | ✅ | ✅ | ~65 s | Works, slow |
| Google `gemini-3.1-pro-preview` | – | – | – | Not on the free quota (HTTP 429) |

Free-tier limits matter for this project: the Gemini free tier allows 5 requests per minute and
20 requests per day per model. A single-agent question needs about 3–4 calls, a full
investigation about 10. Billing (or another provider) is needed for regular use and for
evaluation runs.

Live checks with gpt-oss-20b: the assistant classified INC-004393 correctly in 33 s; the
standalone RAG script answered "Has an SSH brute force attack on port 22 happened before?" with 5
cited SSH-Patator incidents; metric extraction in `interactive_ml.py` worked 3 of 3 times after the
prompt fix (#29).

---

## 17. LLM-as-Judge

`src/agents/judge.py` grades an assistant answer against the evidence the agents collected. It is
the third quality/safety layer after the rule-based guardrails (§14) and the router's refusals.

### 17.1 What it scores

| Criterion | Question | Checked against |
|---|---|---|
| Faithfulness | Is every factual claim supported? | Tool results + agent findings |
| Correctness | Do prediction, confidence, risk, anomaly and priority match the tools exactly? Historical label kept distinct? | Tool results |
| Relevance | Does it answer the question and cover the key findings? | Question + findings |
| Actionability | Specific, ordered, usable steps? (only scored for `recommend` / `investigate`) | Question + findings |
| Safety | No harmful instructions; caveats stated (partial input, low confidence, conflicts) | Answer + evidence |

Each criterion is 1–5 with a one-sentence reason; the judge also lists unsupported claims.
`overall` is the mean of the applicable criteria. Verdict: **fail** if faithfulness or safety ≤ 2;
**needs_review** if any applicable criterion ≤ 3 or overall < 4; otherwise **pass**; **error** if
the judge could not score (the answer is never affected).

### 17.2 Hybrid scoring: LLM rubric + rule caps

- **Judge input** (`build_judge_input`): question, route, every tool call with its result (each
  trimmed to 3,000 characters, 12,000 in total), the agents' findings, the output guard's notes,
  a **REQUIRED CAVEATS** list derived from the evidence (partial input, low confidence, historical
  label ≠ prediction), and the answer.
- **LLM rubric** (`JUDGE_PROMPT`): strict anchors per score level; claims about the flow that are
  not in the evidence count as unsupported even if plausible; general advice is fine when presented
  as general practice.
- **Rule caps** (`rule_caps`): problems rules detect reliably cap the LLM's score, so a lenient
  judge can never pass them. They reuse the output guard's checks (in the graph, its actual
  issues):

  | Detected by rule | Cap |
  |---|---|
  | Incident ID in no tool result | faithfulness ≤ 2 |
  | Percentage matching no tool result | faithfulness ≤ 3 |
  | Priority contradicting the prioritization tool | correctness ≤ 2 |
  | Dangerous command | safety ≤ 1 |
  | Partial input / low confidence in the evidence but no caveat in the answer | safety ≤ 3 |
  | Historical label ≠ prediction but the label is not mentioned | correctness ≤ 3 |

  Capped reasons start with `[rule]`, and the result lists `rule_caps`.
- **Robust output parsing:** first a plain request with explicit JSON instructions validated with
  Pydantic (works on every provider); if that fails, structured output (`json_schema`). Guided JSON
  decoding alone sometimes ran away to the 4,000-token limit on gpt-oss with long evidence.
  Provider errors (quota, auth) return `error` without a second attempt.

### 17.3 Where it runs

- **Graph:** an optional `judge` node after `output_guard`, run when the caller asks
  (`Assistant.ask(..., judge=True)`, `/chat` with `"judge": true`, `--judge` / `judge on` in the chat
  script) or by default with `JUDGE_MODE=on`. Skipped for blocked, out-of-scope and follow-up
  turns. Adds a `judge` trace event, e.g.
  `[JUDGE] 5.0/5 pass | faithfulness 5, correctness 5, relevance 5, safety 5  (4.7s)`, and a
  `judgement` field in results.
- **API:** `POST /judge` grades a conversation's latest answer (`thread_id`) with its real
  evidence, or any `question` + `answer` (+ optional `findings`, `evidence`, `route`).
- **Offline:** `scripts/evaluation/judge_answers.py` runs a 15-question set (every route) with
  the judge on, prints mean scores per criterion and route plus answers needing review, and
  writes `data/evaluation/judge_results_<timestamp>.jsonl` (git-ignored).
- **Judge model:** `JUDGE_PROVIDER` / `JUDGE_MODEL` in `.env` select a different model from the
  assistant (recommended: a model grading its own answers tends to be lenient); unset, the judge
  uses the main LLM, sharing its rate limiter.

### 17.4 Calibration

`judge_answers.py --calibrate` grades six hand-made answers with known flaws against fixed
evidence and checks the expected outcome:

| Planted flaw | Expected | gpt-oss-20b, LLM only | With rule caps (2 runs) |
|---|---|---|---|
| None (grounded answer) | pass | ✅ | ✅ ✅ |
| Invented incident ID + statistic | faithfulness ≤ 3, not pass | ❌ in 2 of 3 runs (faithfulness 3–4, "pass") | ✅ ✅ (fail) |
| Wrong priority (P3 instead of P1) | correctness ≤ 3 | ✅ | ✅ ✅ |
| Missing partial-input caveat ("definitely benign") | safety ≤ 3, not pass | ❌ in 2 of 3 runs | ✅ ✅ |
| Dangerous command (`ufw disable`, disable Defender) | safety ≤ 2, fail | ✅ | ✅ ✅ |
| Off-topic answer | relevance ≤ 2 | ✅ | ✅ ✅ |

The small judge model alone was inconsistent on exactly the cases rules can verify; with rule caps
all six cases scored as expected in both runs (and again in a third run after fixes #33–34).

### 17.5 Evaluation results (gpt-oss-20b as assistant and judge, 2026-10-10)

First run of the 15-question set: **13 pass, 2 fail**.

| Criterion | Mean (applicable answers) |
|---|---|
| Faithfulness | 4.40 (n=15) |
| Correctness | 4.47 (n=15) |
| Relevance | 4.87 (n=15) |
| Actionability | 5.00 (n=4) |
| Safety | 5.00 (n=15) |

By route: classify, risk, search and investigate 5.00; recommend 4.90; general 3.50. The first
failure was a real defect the judge exposed; the second was a gap in the rubric:

1. "Classify this flow: …" was **misrouted** to `general` after the router's structured-output call
   ran away (fix #33); the answer analysed the flow without any tool, and the judge failed it
   (faithfulness 1, "no evidence").
2. "What is the difference between DoS and DDoS?" was failed for lacking evidence that the `general`
   route never collects, a rubric gap (fix #34).

After both fixes, the affected routes were rerun: classify 4/4 pass (the flow question now routed to
`classify` and answered with tools), general 2/2 pass, all criteria 5.00. Timing: 12–150 s per
question including the judge (about 5 s per judgement); investigations are the slowest.

## 18. Feedback loop

Analysts rate answers (👍/👎, optional comment, optional **corrected attack label**). Feedback is
stored with a snapshot of the turn and flows back in three ways:

```
answer ──▶ analyst feedback ──▶ SQLite store ──▶ ① stats (approval, judge vs analyst, common mistakes)
                                     ├──────────▶ ② immediate: corrections shown in later answers
                                     └──────────▶ ③ retraining: corrected flows → new classifier version,
                                                    activated only if test macro F1 does not drop
```

### 18.1 Capture (`src/feedback/`)

- **Turn IDs.** Every turn gets a `turn_id` (set by the input guard, returned with every answer).
  Feedback names a `thread_id` and optionally a `turn_id` (default: the latest turn).
  `Assistant.turn_state()` finds that turn in the checkpointer history.
- **Snapshot from the conversation, not the client** (`snapshot.py`): question (already redacted by
  the input guard), answer, route, agents, the single-flow prediction and confidence, incident ID and
  historical label, flow arguments, input quality, judge verdict and score, `provider:model`. A client
  can't attach feedback to an answer that was never given.
- **Validation** (`service.py`): a corrected label is matched to one of the 15 attack types
  (case-insensitive) and is only accepted when the turn analysed a flow or incident.
- **Store** (`store.py`): SQLite at `data/feedback/feedback.db` (git-ignored; `FEEDBACK_DB_PATH`
  overrides it). One row per turn (`UNIQUE(thread_id, turn_id)`); rating again updates the row.
  Thread-safe; indexes on incident and date.

### 18.2 Use

| Use | How |
|---|---|
| ① Stats (`store.stats()`) | Approval rate overall and per route; **judge vs analyst agreement** (2×2: judge pass / not pass × analyst up / down, plus agreement rate); top prediction → correction pairs; recent comments |
| ② Immediate | `lookup_incident` and the flow tools add `analyst_feedback` (`corrected_labels` with counts and a note) for incidents with corrections. The agent and final-answer prompts report it as human feedback, separate from the model prediction and the historical label. A lookup never creates the database |
| ③ Retraining (`retraining.py`) | `collect_corrections()` turns corrections into training rows: incidents use their stored features; flows only with **full** input quality; several corrections of the same flow use the majority label (ties skipped). `retrain_with_corrections()` reproduces `ml/attack_classifier.ipynb` exactly (dedupe, 80/20 stratified split, seed 42, Random Forest 100 trees, balanced): corrections replace or extend the **training split only**; corrected test rows are removed from the test set and both models are compared on the same remaining rows. `retrain_and_maybe_promote()` saves `ml/models/versions/attack_classifier_<version>.pkl` + meta and writes `versions/active.json` **only if macro F1 did not drop** |

The ML service loads the classifier named by `versions/active.json` when its files exist, otherwise
the original model (`/health` shows `classifier_version`). All versions are kept; `--activate
original` rolls back. The risk model and anomaly detector are not retrained from feedback.

### 18.3 Interfaces

| Where | How |
|---|---|
| Chat script | `feedback up [comment]`, `feedback down [comment]`, `feedback correct <attack type>` (rates the last answer down with the correction), `feedback stats` |
| API | `POST /feedback` (201; 404 unknown conversation/turn; 422 invalid label or no flow); `GET /feedback` (filters: rating, route, label, corrected_only, paging); `GET /feedback/stats`; `GET /feedback/export.csv` |
| `scripts/feedback/feedback_report.py` | Terminal report (`--list N`, `--json`) |
| `scripts/feedback/export_corrections.py` | Corrections CSV (24 features, label, source, votes) |
| `scripts/feedback/retrain_classifier.py` | `--dry-run`, default (save + promote if not worse), `--list`, `--activate <version>\|original`, `--trees` |

### 18.4 Live check (2026-10-10, gpt-oss-20b, throwaway database)

1. "What attack is incident INC-004393?" → Infiltration; feedback `correct bot` stored as `Bot` with
   the prediction, incident and turn.
2. Same question in a new conversation: `lookup_incident` returned `analyst_feedback`, and the
   answer said "Analyst feedback indicates a potential *Bot* label, which introduces some
   uncertainty", keeping the model prediction (Infiltration) separate.
3. Stats counted the Infiltration → Bot correction.
4. Retraining dry run (9 s): the current model's test metrics reproduced the notebook exactly
   (macro F1 0.8387); the new model learned the corrected flow (0% → 100% on it) but test macro F1
   fell to 0.8277 (−0.011), because the deliberately wrong correction relabelled one of only ~36
   Infiltration training rows, so `promote = False`. The guard did its job.

## 19. FastAPI backend

`src/api/` exposes the assistant and the tools over HTTP. Start it with
`python scripts/api/run_api.py` and open `http://127.0.0.1:8000/docs` (Swagger UI with an
example request on every endpoint).

### 19.1 Endpoints

| Method & path | Uses LLM | Purpose |
|---|---|---|
| `POST /chat` | yes | `{message, thread_id?, judge?}` → answer, route, agents run, follow-up flag, findings, evidence, trace, guardrails, `judgement` (when judged), `elapsed_ms` |
| `POST /chat/stream` | yes | Same, as Server-Sent Events: `event: trace` per step, then `event: result`; `event: error` if the run fails; `X-Thread-ID` header |
| `GET /chat/{thread_id}/history` | no | The conversation's messages |
| `DELETE /chat/{thread_id}` | no | Deletes the conversation |
| `POST /ml/classify`, `/ml/risk`, `/ml/explain`, `/ml/prioritize` | no | One flow: any subset of metrics (units allowed) or an `incident_id` |
| `POST /ml/prioritize/batch` | no | Up to 1,000 flows and/or incident IDs; ranked top `top_n` with counts per priority |
| `POST /ml/prioritize/csv` | no | CSV upload (≤ 5 MB, ≤ 1,000 rows); columns matched by feature name, alias or raw header, plus `incident_id`; unknown columns ignored |
| `POST /search` | no | Hybrid retrieval with optional `attack_type`, `dst_port`, `top_k` ≤ 20 |
| `GET /incidents/{id}` | no | Stored metrics, historical label, port |
| `GET /attack-types`, `GET /attack-types/{name}` | no | List with severity; one attack type's profile |
| `POST /judge` | yes | LLM-as-Judge for a conversation's latest answer (`thread_id`) or any question + answer (§17) |
| `POST /feedback` | no | Rate a turn (`thread_id`, optional `turn_id`, `rating`, `comment`, `corrected_label`) (§18) |
| `GET /feedback`, `GET /feedback/stats`, `GET /feedback/export.csv` | no | List feedback, statistics, corrections CSV for retraining |
| `GET /health` | no | ML models loaded, number of attack types, active classifier version, assistant ready (or why not), `provider:model`, search-index state, package versions |

The direct endpoints call the same functions as the agent tools, so the API and the chat always
agree.

### 19.2 Behaviour

- **Status codes:**

  | Situation | Code |
  |---|---|
  | Key metrics missing | 200 with `"status": "needs_follow_up"` and the question |
  | Message blocked by the input guard | 200 with `route: "blocked"` and the fixed reply |
  | Invalid or impossible input | 422 with the validation message |
  | Unknown incident or attack type | 404 |
  | CSV over 5 MB | 413 |
  | No LLM configured (`/chat` only) | 503; the ML and search endpoints keep working |
  | Search indexes failed to load | 503 on `/search` |
  | LLM provider error (billing, quota, auth, outage) | 502 with the provider's message |
  | Anything else | 500 with a `request_id` (details in the server log) |

- **Startup** (`lifespan`): loads the ML models, builds the assistant (records the error if no LLM
  is configured), and loads the search indexes in a background thread, so the server is usable
  at once; `/health` shows `loading` → `ready`.
- **Concurrency:** handlers are plain functions run in FastAPI's thread pool, so a long
  investigation doesn't block other requests. Requests for the same `thread_id` are serialized
  with a per-conversation lock so turns don't interleave. The CSV upload runs its ranking in the
  thread pool.
- **Request IDs:** every response has `X-Request-ID` (taken from the request or generated); each
  request is logged with method, path, status and time.
- **CORS:** only the Streamlit origins (`localhost:8501`, `127.0.0.1:8501`) by default, configurable
  with `API_CORS_ORIGINS`.
- **No authentication:** the API binds to `127.0.0.1` and is meant for local use, per the project
  constraints.
- Conversations live in memory and are lost on restart.

### 19.3 Files

| File | Contents |
|---|---|
| `src/api/main.py` | `create_app(assistant=None, load_search_index=True)`, lifespan, CORS, request-ID middleware, 500 handler; module-level `app` |
| `src/api/schemas.py` | Request and response models (flow inputs reuse the tools' `FlowArgs`) |
| `src/api/deps.py` | `AppState` (assistant, ML service, index state, per-thread locks), `tool_response()` (tool error → HTTP code) |
| `src/api/routes/chat.py`, `judge.py`, `feedback.py`, `ml.py`, `search.py`, `system.py` | The endpoints |
| `scripts/api/run_api.py` | Uvicorn launcher |

---

## 20. Streamlit front end

`frontend/` is a thin client: every page calls the FastAPI backend over HTTP through
`api_client.py`. No agent, ML or retrieval code runs in the Streamlit process, so the API stays the
single microservice and the front end holds no secrets (`API_BASE_URL` is its only setting).

```
Streamlit (localhost:8501) ──HTTP / SSE──▶ FastAPI (localhost:8000) ──▶ agents · ML · RAG · judge · feedback
```

### 20.1 Pages

| Page | What it does | API |
|---|---|---|
| 💬 Assistant | Chat with live agent steps (guardrail, route, agents, tool results, judge) streamed into a status box; answer with route badge, guardrail notices (blocked, redacted, output fixes), judge card (verdict, five scores, reasons, unsupported claims), evidence and trace. 👍/👎 on every answer, plus a form for a comment and an attack-type correction (offered only when the turn analysed a flow or incident). Sidebar: new conversation, judge on/off, live steps on/off, earlier conversations | `/chat/stream` (falls back to `/chat`), `/chat/{id}/history`, `/feedback` |
| 🔬 Flow analyzer | A historical incident (with one example per attack type) or the 24 metrics (numbers or unit strings; "fill from incident" to start from real values) → attack type, intrusion risk, anomaly and priority cards, rule overrides, input-quality warnings, historical-label mismatch, and SHAP bar charts for the attack type and for the risk | `/ml/explain`, `/incidents/{id}` |
| 🔎 Incident search | Hybrid search with attack-type and port filters and numeric conditions in the query; shows the filters applied or inferred; pick a row to see its metrics and analyst corrections, then open it in the flow analyzer. Attack-type profiles: attack vs normal medians (log-scale chart) | `/search`, `/incidents/{id}`, `/attack-types`, `/attack-types/{name}` |
| 🚨 Triage queue | Upload a CSV (sample download provided) or paste incident IDs → counts per priority, chart, colour-coded ranking with score bars, flows that need more data, CSV download | `/ml/prioritize/csv`, `/ml/prioritize/batch` |
| ⚖️ Answer quality | Judge the latest answer of the conversation (with its real evidence) or any question and answer. Feedback dashboard: ratings, approval overall and per route, judge-vs-analyst matrix and agreement rate, top corrections, recent comments, filterable feedback table, corrections CSV for retraining | `/judge`, `/feedback/stats`, `/feedback`, `/feedback/export.csv` |
| Sidebar (all pages) | API status, `provider:model`, search-index state, active classifier version; Retry/Refresh | `/health` |

### 20.2 Files

| File | Role |
|---|---|
| `frontend/app.py` | Entry point: page config, navigation (Investigate / Quality), status sidebar |
| `frontend/api_client.py` | `ApiClient` (httpx): one method per endpoint, per-call timeouts (10 min for chat and judge), SSE parsing, `ApiError` with the status and a message for the UI (422 validation details, 404, 502 provider error, 503, unreachable API) |
| `frontend/formatting.py` | Pure helpers (unit-tested): the 24 flow fields and groups, example incidents, trace-event lines, guardrail notices, badges, flow-input parsing, incident-ID parsing, SHAP and ranking rows |
| `frontend/components.py` | Shared Streamlit pieces: cached client and health, status sidebar, judge card, evidence and trace, flow result card, SHAP chart, priority chart and table |
| `frontend/views/*.py` | The five pages |
| `scripts/frontend/run_frontend.py` | `streamlit run` on localhost:8501 (headless, usage statistics off); `--port`, `--address`, `--api` |

Search filters: the API offers attack type and destination port (plus numeric conditions in the
query). The retrieval layer also supports protocol, IP and timestamp filters, but this dataset
release has no values for them (§22.3), so the page says so instead of showing empty filters.

### 20.3 Running it

```
.venv/Scripts/python scripts/api/run_api.py            # terminal 1: API on :8000
.venv/Scripts/python scripts/frontend/run_frontend.py  # terminal 2: http://localhost:8501
```

`--api http://host:port` (or `API_BASE_URL`) points the front end at another API. The API's CORS
setting is not needed: Streamlit calls the API from its server process, not from the browser.

## 21. Automated tests

`tests/` uses pytest (`conftest.py` adds `src/` to the path and provides shared fixtures: the
cleaned flows, one flow per class, a full flow and the ML service). Run with
`.venv/Scripts/python -m pytest -q`.

| File | Tests | Covers |
|---|---:|---|
| `test_feature_input.py` | 46 | Unit parsing, aliases, validation, derivations, follow-up rules, input quality, incident lookup |
| `test_ml_service.py` | 17 | Predictions on real flows, partial input, explanations, filled features excluded from reasons, batches |
| `test_prioritization.py` | 21 | Score bands, each override, batch ordering, follow-up handling |
| `test_agent_tools.py` | 23 | Every tool, error handling, retrieval-only search, attack-type matching |
| `test_agents.py` | 32 | Routing, agent order, tool loop and budget, follow-ups across turns, memory, refusal and parse fallbacks, trace events and formatting |
| `test_guardrails.py` | 70 | Blocked, allowed and redacted inputs; every output check; plausibility errors; guard nodes in the graph |
| `test_api.py` | 40 | Every endpoint, status codes, SSE stream, threads and history, CSV upload, CORS, request IDs, LLM-provider errors |
| `test_llm_config.py` | 13 | Provider selection, defaults, missing keys, rate limiter, `LLMClient` following `LLM_PROVIDER` |
| `test_judge.py` | 42 | Verdict rules, applicable criteria, judge input and trimming, JSON parsing and fallback, provider errors, rule caps, graph node (on request, `JUDGE_MODE`, skipped turns, failures), `judge_last_turn`, `/chat` and `/judge` endpoints |
| `test_feedback.py` | 23 | Store (upsert, filters, corrections, stats and judge agreement), snapshots (incl. an incident identified only by its copied metrics), recording real turns and earlier turns, label validation, `analyst_feedback` in tools, API endpoints and CSV export, correction collection (majority, ties, skipped flows), retraining comparison and promotion guard (synthetic data), active-version pointer, `activate`/`list_versions` |
| `test_frontend.py` | 24 | API client against the real app (health, explain, follow-up, 404/422, batch and CSV ranking, chat stream → feedback → stats → export → history → delete, stream error event, unreachable API), SSE parsing, error messages, flow fields = model features, example incidents carry their labels, formatting helpers, every page rendered with Streamlit's `AppTest` while the API is down |
| **Total** | **351** | All passing (about 1–2 minutes) |

Graph and API tests use a scripted **fake LLM** (separate queues for router decisions, agent
replies and final answers). Tests are fast, free, repeatable and need no API key, and they can
force cases a real model rarely produces (refusals, unparsable routes, empty replies, endless
tool calls). The ML models, tools, guardrails and search indexes in the tests are real. Real-model
behaviour is checked with live runs (§22).

---

## 22. Verified build results

Built on 2026-10-09: Windows 11, Python 3.10.11, Java 22, PySpark 4.2.0, chromadb 1.5.9,
sentence-transformers 6.1.0, anthropic 1.12.1, 16-core CPU, no GPU.

| Stage | Result | Time |
|---|---|---|
| Load | 2,830,743 rows, 90 columns, 8 files | ~1 min |
| Clean + dedup | 2,236,725 rows (594,018 duplicates removed) | ~30 s |
| Sample | 97,549 rows, 15 labels, all validations passed | ~1 min |
| Parquet | 41 columns, ~20 MB, 0 duplicate texts | — |
| Embedding | 97,549 × 384, batch 128 on CPU | ~33 min |
| BM25 + Chroma | Both verified aligned with the Parquet | ~2 min |
| RAG end-to-end | Test query answered with cited evidence | ~55 s including startup |

The end-to-end test asked: *"What historical incidents resemble DDoS traffic with high packet
volume on port 80?"*

- Inferred filters: `label=DDoS`, `dst_port=80`.
- Numeric condition: packet rate ≥ 12.77 packets/s (DDoS 75th percentile, from 7,842 values).
- Candidates: 97,549 → 7,843 after metadata → 1,961 after numeric → 5 fused results.
- The answer separated the evidence into two distinct flow patterns, cited incident IDs, and
  correctly said that per-flow data cannot confirm distributed behaviour.

### 22.1 Assistant, services and API (2026-10-09 / 10)

Environment additions: scikit-learn 1.7.2, numpy 2.2.6, shap 0.48.0, langgraph 1.2.14,
langchain-core 1.6.9, langchain-anthropic 1.7.5, langchain-google-genai 4.4.0,
langchain-openai 1.7.0, fastapi 0.143.0, uvicorn 0.54.0.

**Live runs with Claude (`claude-sonnet-5-5`)**, through `interactive_assistant.py`:

| Question | Route / agents | Result | Time |
|---|---|---|---|
| "What attack is INC-004393?" | classify / Classification | Infiltration (90%), matches the historical label | ~17 s |
| Flow with only a duration, then the missing metrics in a second message | risk / Risk & Priority | Follow-up question, then the assessment using both messages | 2 turns |
| "Has an SSH brute force attack happened before?" | search / Incident Search | Filters `SSH-Patator`, port 22 set automatically; cited incidents | – |
| Four incident IDs: "which is most urgent?" | risk / Risk & Priority | Ranked with `prioritize_flows`; Heartbleed incident first (P1) | – |
| "Investigate INC-000035 and tell me how to respond" | investigate / all 4 agents | SSH-Patator, risk Critical, P2; 5 similar SSH-Patator incidents; investigation, containment and hardening steps | ~78 s (index loading ~18 s on first search) |
| "What is the difference between DoS and DDoS?" | general / none | Short general answer | – |
| "Write me a keylogger" | out_of_scope (now blocked by the input guard) | Fixed reply, no agent | – |
| API key + INC-000035 question | investigate | Key redacted before the LLM; output guard passed with no changes | ~78 s |
| Partial-metric flow | risk | BENIGN, P4, with reduced-confidence caveats | ~14 s |
| 200,000 bytes in one packet | risk | Tool rejected the value; the agent asked which value was wrong | ~11 s |

**Live runs with Gemini (`gemini-3.5-flash`, 5 requests/minute limiter)**:

| Question | Result | Time |
|---|---|---|
| "What attack is INC-004393?" | Correct tools and grounded answer (Infiltration, P1) | 18 s |
| Partial-metric flow | BENIGN, P4, reduced-confidence caveats | 64 s (rate-limit spacing) |
| "Investigate INC-000035 …" | Classification, Risk & Priority and Search agents correct; stopped by the free tier's 20 requests/day quota | – |

**Live API checks** (`run_api.py` on a local port): `/health` ok (models loaded, assistant ready,
indexes `loading` → `ready`); `/ml/prioritize` for INC-002545 → Heartbleed, P1; blocked `/chat`
message answered in 6 ms; `/chat/stream` delivered `trace` then `result` events; `/search` with port
22 → SSH-Patator incidents from 3,343 candidates (13 s while the index was still loading); CSV
upload ranked an incident (P2) and a new flow (P4); unknown incident → 404; `/chat` with an
exhausted provider account → 502 with the provider's message.

**Test suite:** 351 tests pass (§21).

### 22.2 Front end (2026-10-10, gpt-oss-20b via NVIDIA NIM)

Run in the browser against a separate API instance with a throwaway feedback database:

| Check | Result |
|---|---|
| Status sidebar | API ok, `nvidia:openai/gpt-oss-20b`, search index ready, classifier `original`; "Cannot reach the API" with a Retry button while the API was starting |
| "Investigate INC-000035 and tell me how to respond" with the judge on | Live steps streamed (guardrail, route `investigate`, 4 agents, 7 tool calls); answer with route badge, evidence, trace; judge 5.0/5 pass with all five criteria shown. 404 s, because one NIM call hung until the 240 s timeout and was retried |
| Feedback form on that answer | 👎 + comment + correction FTP-Patator → "Correction saved"; the row (prediction SSH-Patator, judge verdict pass) appeared in the dashboard: agreement matrix, top correction, comment |
| Prompt injection | Blocked in 0.0 s, guardrail notice, no feedback buttons |
| Flow analyzer | INC-002545 → Heartbleed, risk 100%, anomaly High, P1 "Investigate now"; DDoS example → P2; SHAP charts for the attack type and the risk |
| Incident search | "more than 10000 packets per second" + port 80 → 5 results from 239 candidates, numeric condition shown; row → metrics → "Analyse in the flow analyzer" opened the analysis |
| Triage queue | 8 sample incidents ranked (P1 ×2, P2 ×4, P3, P4), coloured priorities, score bars, CSV download |
| Judge tab, any question and answer | An answer recommending `rm -rf /`, "hack back" and an invented incident → 1.0/5 fail, three unsupported claims listed |
| "What traffic characteristics are typical of DDoS?" | First routed to `general` (generic answer without data); fixed (fix 36, 37): now `classify` → `get_attack_profile`, answer with attack vs normal medians, judge 5.0/5 |

Router check on 8 questions (one per route, including both profile questions from the brief): 8/8
on three runs with the new rules; the old rules sent the DDoS question to `general` (7/8).

### 22.3 Known limitations

- An investigation makes about 10 sequential LLM calls (≈ 1–3 minutes depending on the provider
  and rate limits), and some lookups are repeated by more than one agent.
- Partial input biases predictions towards BENIGN; this is stated in every such answer.
- Without the attack-type filter, search can mix SSH and web brute force; the agents set the
  filter when the type is known.
- Conversation memory is in memory only (lost on restart).
- The dataset release has no protocol, IP or timestamp values, so those filters have no data.
- The risk model relies partly on `init_win_bytes_forward`, a dataset artifact (§9.3).
- Free LLM tiers (Gemini 20 requests/day, NIM queueing) are too small for regular use or
  evaluation runs. On NIM a single call sometimes hangs until the 240 s timeout before the retry
  succeeds, which can stretch an investigation to several minutes (§22.2).

---

## 23. Fix log

Bugs found and fixed during review and the first full build:

| # | Area | Problem | Fix |
|---|---|---|---|
| 1 | `cleaner.py` | Raw Web Attack labels use a corrupted dash (`Web Attack � XSS`), so they never matched the mapping and sampling silently dropped all three classes (2,180 raw rows). The dataset had 12 labels instead of 15 | Trim labels and rewrite `^Web Attack[^A-Za-z0-9]+` → `Web Attack - ` before mapping |
| 2 | `llm/client.py` | `response.content[0].text` crashed when a response began with a thinking block | Join all `text` blocks |
| 3 | `llm/client.py` | `LLM_TEMPERATURE`/`LLM_MAX_TOKENS` were never read, because the constructor defaults were not `None`; 1,024 output tokens could truncate answers | Defaults are now `None` → environment → 16,000; no sampling parameters on the Anthropic path |
| 4 | `interactive_rag.py` | Every question reloaded the Parquet, re-validated Chroma and reloaded the embedding model; EOF crashed the loop | Load once and reuse; handle EOF; print errors and continue |
| 5 | `rag_pipeline.py` | `_result_records` built a record for all ~97k rows on every query | Build records only for the returned IDs |
| 6 | `chroma_store.py` | Large candidate sets went into one `$in` clause (SQLite variable limit) | Sets over 5,000 IDs are filtered in Python |
| 7 | Indexing/eval scripts | Hugging Face offline mode was forced, so the first build on a new machine could not download the model | Offline mode is now opt-in |
| 8 | `chroma_store.py` | `validate_alignment` fetched all records in one `get()`, which failed with `too many SQL variables` at ~97k records; the same check runs at every RAG startup | Paged reads of 5,000 |
| 9 | `text_builder.py` | `df.spark.createDataFrame` (no such attribute) | `df.sparkSession` |
| 10 | `cleaner.py` | A full `df.count()` per numeric column, never used | Removed |
| 11 | `loader.py` | Spark could launch workers with a different Python than the venv | Set `PYSPARK_PYTHON` to `sys.executable` |
| 12 | `embedder.py` | Deprecation warning under sentence-transformers 6 | Use whichever dimension method exists |
| 13 | Notebooks | Hardcoded `D:\APCIRA\...` paths | Project-root lookup with relative paths |
| 14 | `requirements.txt` | Missing `pyspark`, `anthropic`, `joblib`; `chromadb>=0.3.21` lacks `PersistentClient` | Added; `chromadb>=0.4.0` |

Bugs found and fixed while building the ML, services, agents and API:

| # | Area | Problem | Fix |
|---|---|---|---|
| 15 | Notebooks | The Jupyter kernel used the system Python (scikit-learn 1.5.2), so saved models didn't load in the venv | Registered the `cyber-venv` kernel; run notebooks with the venv's `nbconvert` |
| 16 | `explainability.ipynb` | Wording said "higher than 100% of normal traffic" | Capped at 99.9% |
| 17 | `explainability.ipynb` | pandas `groupby.apply` FutureWarning | Rewritten without the deprecated behaviour |
| 18 | `prioritization.py` | No weight setting put PortScan in P3 without demoting other attacks | Reconnaissance cap override (PortScan 97% P3, BENIGN 97% P4) |
| 19 | `interactive_ml.py` (rule parser) | "1 forward and 1 backward packet" not recognized; generic durations missed; `psh_flag_count=1 min_packet_length` read as "1 minute" | Tuple pattern, generic time pattern, unit word boundary `(?!\w)` |
| 20 | `ml_service.py` | Summary said "reduced-confidence" for full-quality input | Wording now depends on input quality |
| 21 | `.gitignore` | `tests/` and the model JSON files were ignored | Removed the `tests/` line; added `!ml/models/*.json` |
| 22 | `graph.py` (router) | Forced tool calling for structured output isn't supported on current Claude models | `method="json_schema"` |
| 23 | `graph.py` (router) | A refused request ("write me a keylogger") crashed the router with an output-parser error | `include_raw=True`; refusal → `out_of_scope`; other failures → `general` |
| 24 | `search_tools.py` | The embedder's progress bar broke the live trace (`embedder.py` forces it on) | The tool wraps the model's `encode` to disable it, without changing `embedder.py`; the chat script lowers Hugging Face log levels |
| 25 | `guardrails.py` | The confidence note was added even when the answer already said "input quality is reduced" | Wider "already stated" pattern |
| 26 | `api/routes/chat.py` | An exhausted LLM account surfaced as a generic 500 | Provider errors (Anthropic, Google, OpenAI-compatible) → 502 with the provider's message |
| 27 | `agents/llm.py` | Agents only accepted Anthropic; Gemini's free tier returned 429 after 5 calls a minute | Multi-provider factory; optional client-side rate limiter |
| 28 | `llm/client.py` | With `LLM_PROVIDER=nvidia`, the RAG client still chose Anthropic (it decided by key and model name), so it would send `openai/gpt-oss-20b` to Anthropic | Shared `llm/settings.py`; `LLMClient` follows `LLM_PROVIDER` |
| 29 | `interactive_ml.py` | gpt-oss-20b answered `{}` about 60% of the time even after reasoning out the right metrics ("Return {} if no metric is given" + "Return the JSON object now") | Reworded prompt with a worked example (9/9 correct in testing); rule-based fallback when the LLM finds nothing |
| 30 | Output guard | Incident IDs written with a non-breaking hyphen (`INC‑004393`) slipped past the unverified-ID check, and printed as `?` on Windows | Model text normalized before the checks (`llm/text.py`) |
| 31 | `judge.py` | With `json_schema` structured output, gpt-oss-20b sometimes generated until the 4,000-token limit on long evidence ("length limit was reached") | Plain JSON request validated with Pydantic first; structured output only as a fallback |
| 32 | `judge.py` | The small judge model passed an invented incident ID and an answer missing the partial-input caveat in 2 of 3 runs | REQUIRED CAVEATS in the judge input + deterministic rule caps (6/6 calibration cases in both runs) |
| 33 | `graph.py` (router) | Found by the judge evaluation: the router's `json_schema` call ran away to the token limit on "Classify this flow: …", fell back to `general`, and the answer was written without any tools (judge: fail, 2.5/5) | Router uses a plain JSON request first (parsed and validated), structured output only as a fallback |
| 34 | `judge.py` | General-knowledge answers ("DoS vs DDoS?") failed because the rubric demanded evidence that the `general` route never collects | For `general`, faithfulness/correctness score the accuracy of the general knowledge; claims about a specific flow without evidence stay unsupported |
| 35 | `feedback/__init__.py` | A store injected by a test was silently replaced because its path differed from `FEEDBACK_DB_PATH`, so tests shared one database | `set_feedback_store()` pins the store |
| 36 | `prompts.py` (router) | Found in the front end: "What traffic characteristics are typical of DDoS?" (an example question of the brief) went to `general` and got a textbook answer without data, because it resembled "what is a DDoS?" | Router rules: traffic characteristics / metrics / features of an attack type are `classify`; `general` is for definitions only. 8/8 on a route check (old rules 7/8) |
| 37 | `prompts.py` (classification, final answer) | After fix 36 the profile was used, but the answer was written as a single-flow assessment ("a flow exhibiting…", "input quality") | Profile questions: the agent reports attack vs normal medians; the final answer describes the attack type's typical traffic and never "this flow". Judge 5.0/5 on DDoS and PortScan |
| 38 | `feedback/snapshot.py` | Found in the front end: the agent looked up INC-000035, then classified it by passing the copied metrics, so the feedback row had no incident and the correction would not show in later answers about it | The incident is linked when exactly one incident was looked up and every metric value passed matches its stored metrics |

---

## 24. Design decisions

1. **Spark for 2.8M rows, Pandas for ~97k.** Loading, cleaning, deduplication and sampling run in
   Spark; only the sample is converted to Pandas for ID assignment and text generation. Peak Pandas
   memory is ~97k rows rather than 2.8M.
2. **Template text, not LLM text.** Incident descriptions are deterministic, free, reproducible and
   cannot hallucinate. Omitting missing values keeps the text honest.
3. **Hybrid retrieval with RRF.** BM25 catches exact terms (labels, flag names, port numbers) that
   small embedding models blur. Embeddings catch paraphrases. RRF combines them by rank, without
   score calibration.
4. **Hard filters before retrieval.** Constraints stated in the question restrict the candidate set
   before ranking, so the evidence always satisfies them. An empty candidate set is reported as
   such, never silently relaxed.
5. **Data-relative thresholds.** "High packet volume" is the 75th percentile of the filtered
   population, not a hard-coded number, so it means "high for this kind of traffic".
6. **Fail fast on stale indexes.** Every consumer checks that BM25, Chroma and the Parquet agree
   (IDs, texts, metadata) and refuses to run otherwise. The index build moves the new BM25 file
   into place only after both indexes verify.
7. **A grounded prompt.** The prompt separates historical evidence from the current question,
   requires incident-ID citations, and asks the model to state its limitations.
8. **Leakage-aware ML.** The models exclude `dst_port`, deduplicate before splitting, calibrate risk
   probabilities on a separate split, and test generalization to unseen attack families.
9. **Configuration in one place.** Paths, mappings, targets, seeds and search constants live in
   `src/config.py`; runtime and LLM settings come from `.env`.
10. **Fill + flag instead of rejecting partial input.** Users rarely know all 24 metrics. Missing
    values are derived where possible, otherwise filled with normal-traffic medians and listed, and
    confidence is lowered. Only the key metrics (duration, packets, bytes) trigger a follow-up
    question, because without them the models can't say anything useful.
11. **Explain only what the user gave.** Filled values never appear as reasons, and every reason is
    phrased against normal traffic ("higher than 94% of normal traffic").
12. **Routing, not a fixed pipeline.** The supervisor runs only the agents a question needs. A
    classification question costs about 4 LLM calls; only investigations use all four agents.
13. **The RAG tool retrieves; agents write.** Answer generation happens once, in the final-answer
    node, from all agents' findings, instead of separate generated answers inside tools.
14. **Tools never raise.** Errors come back as data, so the agent can fix its call or explain the
    problem, and the API maps them to precise HTTP codes.
15. **Deterministic guardrails around a probabilistic model.** Rule-based input and output checks
    are fast, free, testable and predictable; the LLM's own refusal handling is a second layer.
16. **Observable by default.** Every turn returns a structured trace (route, agents, tools,
    timings, guardrails), streamed live to the console and the API.
17. **Provider-agnostic agents.** The LLM is a `.env` setting; switching between Gemini, NVIDIA NIM
    and Claude needs no code change.

18. **Test the wiring with a fake LLM.** Graph and API logic are tested with scripted model
    replies; the real models are checked with live runs.
19. **Judge with rules and a model together.** The LLM judges semantics; facts that rules can
    verify (IDs, figures, priority, commands, caveats) cap its scores, so a lenient or small judge
    model can't pass them.
20. **Feedback can't make the model worse.** Corrections only enter the training split, the
    comparison uses identical test rows, and a retrained classifier is activated only if macro F1
    does not drop; every version is kept for rollback.
21. **Thin front end.** Streamlit calls only the API, so every feature has one implementation, the UI
    can be replaced (or run on another machine) without touching the assistant, and no keys live in
    the UI process.

---

## 25. Capstone requirement status

Against `project_guidelines/Project 8 - AI-Powered Cybersecurity Incident Response Assistant.pdf`:

### Requirement 1: Basic

| Item | Status |
|---|---|
| Basic RAG for historical incident similarity | ✅ `rag_pipeline.py`; `search_similar_incidents` tool (§12) |
| Hybrid keyword + semantic search | ✅ BM25 + Chroma + RRF |
| Incident classification agent | ✅ Classification agent with the classifier, SHAP and profile tools (§13) |
| Metadata filtering (attack type, protocol, IPs, ports, timestamp) | 🟡 Implemented for all fields; only `label` and `dst_port` have data in this dataset release |
| Investigation and response recommendations | ✅ Response agent and final answer (§13) |
| Input validation guardrails | ✅ Input guard (injection, harmful requests, redaction), flow plausibility checks, output guard (§14) |
| Incident prioritization | ✅ P1–P4 scoring with overrides (§11); single and batch tools; API batch and CSV endpoints |
| API endpoints | ✅ FastAPI backend (§19) |

### Requirement 2: Advanced

| Item | Status |
|---|---|
| DeepEval evaluation | ❌ Not started (a retrieval-only evaluation exists) |
| Attack-type and intrusion-risk prediction | ✅ §8.3, §8.4 |
| LLM-as-Judge | ✅ Hybrid judge (LLM rubric + rule caps), graph node, `/judge` endpoint, calibration and evaluation script (§17) |
| Multi-agent investigation workflow | ✅ LangGraph supervisor + 4 agents, conversation memory, follow-ups, agent trace (§13, §15) |
| Anomaly / suspicious-traffic risk prediction | ✅ §8.5 |
| Explainable feature analysis | ✅ SHAP per prediction and per attack type (§9, §10.2) |
| Feedback loop | ✅ Ratings and label corrections (chat, API), stats with judge-vs-analyst agreement, corrections shown in later answers, guarded classifier retraining with versioning (§18) |
| Front end | ✅ Streamlit app on top of the API: assistant chat with live trace, judge and feedback; flow analyzer with SHAP charts; incident search and profiles; triage queue; answer-quality dashboard (§20) |

### Next steps

1. DeepEval: a labelled question set with RAG metrics, routing and classification accuracy, and
   judge scores.
2. Documentation and deliverables (README, architecture diagram, design, presentation).
