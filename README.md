# APCIRA - AI-Powered Cybersecurity Incident Response Assistant

> Processing CIC-IDS2017 dataset for cybersecurity incident analysis with semantic search and RAG.

## 🚀 Quick Start

```bash
# 1. Prepare raw data (place 8 CSV files in MachineLearningCVE/)
# 2. Run data preparation
python scripts/data/prepare_incident_data.py

# 3. Build search indexes
python scripts/indexing/build_search_indexes.py

# 4. Run interactive search
python scripts/search/interactive_hybrid_search.py

# Optional: run a single query without the prompt
python scripts/search/interactive_hybrid_search.py "DDoS attack high packet volume"

# Run RAG interactively; enter any custom query at the prompt
python scripts/search/interactive_rag.py

# Run one custom RAG query and exit
python scripts/search/interactive_rag.py "What historical incidents resemble high-volume DDoS traffic?"

# Run a RAG query with metadata filters
python scripts/search/interactive_rag.py --filter label=PortScan --filter dst_port=80 --top-k 3 "Find similar historical incidents"

# Optional: inspect normalization and RRF rank contributions
python scripts/search/interactive_hybrid_search.py --debug "PortScan SYN multiple destinations"

# Optional: filter retrieval by metadata (filters combine with AND)
python scripts/search/interactive_hybrid_search.py --filter label=PortScan --filter dst_port=80 --debug "PortScan SYN traffic"

# Optional: run the retrieval evaluation set
python scripts/evaluation/evaluate_hybrid_search.py
```

## 📚 Documentation

All documentation is in the [`docs/`](docs/) folder:

- **[QUICKSTART.md](docs/QUICKSTART.md)** - 3-step quick start guide
- **[DATA_PREP_ARCHITECTURE.md](docs/DATA_PREP_ARCHITECTURE.md)** - Complete technical architecture
- **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)** - Implementation details
- **[METADATA_FILTERING.md](docs/METADATA_FILTERING.md)** - Metadata filter fields and usage
- **[DELIVERABLES.txt](docs/DELIVERABLES.txt)** - Project deliverables checklist

## 📁 Project Structure

```
APCIRA/
├── docs/                              ← Documentation
├── MachineLearningCVE/                ← Raw CSV data (8 files)
├── data/
│   ├── processed/                     ← incidents_100k.parquet (output)
│   └── logs/                          ← Pipeline logs
├── src/
│   ├── config.py                      ← Configuration
│   ├── data/                          ← Modular data pipeline
│   │   ├── loader.py                  ← CSV loading
│   │   ├── cleaner.py                 ← Data cleaning
│   │   ├── sampler.py                 ← Stratified sampling
│   │   ├── text_builder.py            ← Text generation
│   │   └── validator.py               ← Verification
│   └── retrieval/                     ← Search & RAG
├── scripts/
│   ├── data/                          ← Data preparation
│   │   └── prepare_incident_data.py
│   ├── indexing/                      ← Index building
│   │   └── build_search_indexes.py
│   ├── search/                        ← Interactive search
│   │   ├── interactive_hybrid_search.py
│   │   └── interactive_rag.py
│   └── evaluation/                    ← Retrieval evaluation
│       └── evaluate_hybrid_search.py
├── requirements.txt                   ← Dependencies
└── .gitignore                         ← Git ignore rules
```

## 🔄 Pipeline Stages

1. **Load** - Load all 8 CIC-IDS2017 CSV files (~2.8M rows)
2. **Clean** - Normalize, standardize, deduplicate
3. **Sample** - Stratified sampling to ~100K records
4. **Generate** - Create incident descriptions for search
5. **Index** - Build BM25 and embedding indexes
6. **Search** - Query incidents semantically

## ✨ Features

- ✅ Modular, reusable architecture (5 components)
- ✅ PySpark for large-scale processing
- ✅ Stratified sampling (~100K incidents)
- ✅ Hybrid search (BM25 + semantic)
- ✅ RAG with incident context
- ✅ Comprehensive logging and validation
- ✅ Backward compatible with existing pipeline

## 📊 Output

- **incidents_100k.parquet** (~30-50 MB)
  - ~100K stratified incident records
  - 30 columns (label, ML features, incident_text)
  - Ready for search and ML analysis

## 🛠️ Technologies

- **PySpark** - Distributed data processing
- **Pandas** - Final data handling
- **BM25** - Keyword-based search
- **Sentence Transformers** - Semantic embeddings
- **Chroma** - Vector database
- **Parquet** - Columnar storage

## 📖 For More Information

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) to get started immediately.

See [`docs/DATA_PREP_ARCHITECTURE.md`](docs/DATA_PREP_ARCHITECTURE.md) for complete technical details.
