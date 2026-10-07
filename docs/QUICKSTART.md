# Quick Start - Modular Data Preparation Pipeline

## One-Minute Overview

The CIC-IDS2017 data preparation pipeline has been refactored from a monolithic script into **5 reusable modules** organized under `src/data/`:

```
src/data/
├── loader.py       # Load 8 CSV files with PySpark
├── cleaner.py      # Clean, normalize, standardize data  
├── sampler.py      # Stratified sampling to ~100K
├── text_builder.py # Generate natural language descriptions
└── validator.py    # Verify final dataset
```

**Main script:** `scripts/data/prepare_incident_data.py` (orchestration only)

## Quick Start (3 Steps)

### Step 1: Place Raw Data

Download CIC-IDS2017 dataset and place the 8 CSV files in:
```
MachineLearningCVE/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### Step 2: Run Data Preparation

```bash
python scripts/data/prepare_incident_data.py
```

**What happens:**
1. Loads all 8 CSV files (~2.8M rows)
2. Cleans & normalizes data
3. Stratified sampling to ~100K
4. Generates incident descriptions
5. Saves to `data/processed/incidents_100k.parquet`
6. Verifies output

**Time:** ~5-10 minutes  
**Output:** `data/processed/incidents_100k.parquet` (~30-50 MB)  
**Log:** `data/logs/data_prep_YYYYMMDD_HHMMSS.log`

### Step 3: Build Search Indexes

```bash
python scripts/indexing/build_search_indexes.py
```

This will:
- Encode incident texts to 384D embeddings
- Build BM25 index
- Create persistent Chroma vector database

**Time:** ~2-5 minutes (one-time)

To regenerate the data representation and rebuild indexes after retrieval
feature or metadata changes, run:

```bash
python scripts/data/prepare_incident_data.py
python scripts/indexing/build_search_indexes.py --rebuild
```

Run the retrieval evaluation queries and metrics with:

```bash
python scripts/evaluation/evaluate_hybrid_search.py
```

## Then: Run Search

```bash
python scripts/search/interactive_hybrid_search.py
```

or run a single query without entering the interactive prompt:

```bash
python scripts/search/interactive_hybrid_search.py "DDoS attack high packet volume"
```

or

```bash
python scripts/search/interactive_rag.py
```

The RAG command above starts an interactive prompt. Enter a custom question,
then enter `quit` to exit. To run a single custom question directly from the
terminal:

```bash
python scripts/search/interactive_rag.py "What historical incidents resemble high-volume DDoS traffic?"
```

You can also combine a custom query with metadata filters and a context limit:

```bash
python scripts/search/interactive_rag.py --filter label=PortScan --filter dst_port=80 --top-k 3 "Find similar historical incidents"
```

### RAG hard filters for traffic conditions

RAG automatically turns recognized metadata values and numeric traffic
conditions in the query into hard filters before retrieval. For example:

```bash
python scripts/search/interactive_rag.py --debug "DDoS traffic with high packet volume"
python scripts/search/interactive_rag.py --debug "Find DDoS incidents with more than 10000 packets per second"
python scripts/search/interactive_rag.py --debug "DDoS traffic on port 80 with more than 10000 packets/s and duration over 2 seconds"
```

Metadata and numeric conditions combine with AND semantics. The qualitative
phrase `high packet volume` resolves to the 75th percentile of `flow_packets_s`
among records that satisfy the query's metadata filters. Explicit packet-rate,
total-packet, and duration thresholds are applied as written; durations are
converted to the dataset's microsecond unit. Numeric conditions produce eligible
incident IDs before BM25, vector search, and RRF. Thus semantic similarity
cannot introduce records that fail a recognized hard condition. If no incident
matches, the pipeline reports that no matching historical incidents were found
instead of removing the condition.

With `--debug`, the CLI prints resolved numeric thresholds and their source,
candidate counts before and after filtering, retrieval counts, and packet rates
for the final results. The details are also returned in the RAG result as
`numeric_conditions` and `candidate_counts`.

## File Structure

```
APCIRA/
├── MachineLearningCVE/          ← Put raw CSV files here
├── data/
│   ├── processed/
│   │   ├── incidents_100k.parquet       ← Main output
│   │   ├── bm25_index.pkl              ← Search index
│   │   └── chroma/                     ← Vector database
│   └── logs/
│       └── data_prep_*.log              ← Pipeline logs
│
├── src/
│   ├── config.py                        ← Configuration
│   ├── data/                            ← Modular functions
│   │   ├── loader.py       (130 lines)
│   │   ├── cleaner.py      (150 lines)
│   │   ├── sampler.py      (150 lines)
│   │   ├── text_builder.py (200 lines)
│   │   └── validator.py    (170 lines)
│   └── retrieval/                       ← Search & RAG
│       ├── embedder.py
│       ├── bm25_index.py
│       └── chroma_store.py
│
└── scripts/
    ├── prepare_incident_data.py       ← Run this first
    ├── build_search_indexes.py        ← Run this second
    ├── interactive_hybrid_search.py
    └── interactive_rag.py
```

## What Each Module Does

### `loader.py` - Load CSVs
- Loads all 8 CIC-IDS2017 files
- Combines using PySpark (efficient)
- Logs row/column counts
- Verifies required columns

### `cleaner.py` - Clean Data
- Strips whitespace from column names
- Renames to internal schema
- Casts to numeric types
- Handles inf/NaN values
- Standardizes attack labels
- Removes duplicates
- Drops incomplete rows

### `sampler.py` - Stratified Sampling
- Samples to ~100K records
- Preserves class distribution
- Allocates per-class targets:
  - BENIGN: 52,000
  - DoS attacks: ~15,000
  - Port scanning: 7,800
  - Web attacks: ~2,000
  - Rare attacks: Keep all
- Logs before/after distribution

### `text_builder.py` - Generate Text
- Converts network flows to natural language
- Includes attack type, protocol, ports, packets, bytes, duration, flags
- Example:
  ```
  Attack: DoS Hulk. Protocol 6. Source port 52345 destination port 80.
  Forward packets: 1000. Backward packets: 500. Duration: 5000000 microseconds.
  Flags: 1 forward SYN, 1 backward SYN, 500 forward ACK, 500 backward ACK.
  ```

### `validator.py` - Verify Dataset
- Checks incident IDs are unique
- Verifies incident_text is complete
- Reports label distribution
- Detects duplicate rows
- Counts null values
- Prints final schema

## Configuration

Edit `src/config.py` to change:

```python
# Paths
DATA_RAW_DIR = Path("MachineLearningCVE")
DATA_PROCESSED_DIR = Path("data/processed")

# CSV files
RAW_CSV_FILES = [...]  # List of 8 filenames

# Column mapping
RAW_TO_INTERNAL_COLUMNS = {...}  # Raw → internal names

# Sampling targets
SAMPLES_PER_CLASS_100K = {
    "BENIGN": 52000,
    "DoS Hulk": 7800,
    ...
}

# Random seed
RANDOM_SEED = 42
```

## Output Schema

The Parquet file contains:

```python
DataFrame with columns:
  - incident_id (str):       "INC-000001", "INC-000002", ...
  - label (str):             "BENIGN", "DoS Hulk", "DDoS", ...
  - protocol (double):       6, 17, ...
  - src_port (double):       52345, 22, ...
  - dst_port (double):       80, 443, ...
  - flow_duration (double):  5000000, ...
  - total_fwd_packets (double): 1000, ...
  - total_bwd_packets (double): 500, ...
  - total_len_fwd (double):  50000, ...
  - total_len_bwd (double):  30000, ...
  - flow_bytes_s (double):   10000, ...
  - flow_packets_s (double): 300, ...
  - flow_iat_mean (double):  5000, ...
  - packet_length_mean (double): 50, ...
  - fwd_psh_flags (double):  10, ...
  - bwd_psh_flags (double):  10, ...
  - fwd_urg_flags (double):  0, ...
  - bwd_urg_flags (double):  0, ...
  - fwd_rst_flags (double):  0, ...
  - bwd_rst_flags (double):  1, ...
  - fwd_syn_flags (double):  1, ...
  - bwd_syn_flags (double):  1, ...
  - fwd_ack_flags (double):  500, ...
  - bwd_ack_flags (double):  500, ...
  - src_ip (string):         "192.168.1.100", ...
  - dst_ip (string):         "10.0.0.1", ...
  - timestamp (string):      "2018-02-21 09:30:00", ...
  - incident_text (str):     "Attack: DoS Hulk. Protocol 6. ...", ...

Total: ~100,000 rows, ~30 columns
```

## Logging & Output

### Console Output
```
============================================================================
CIC-IDS2017 DATA PREPARATION PIPELINE (Modular)
============================================================================
Output: data/processed/incidents_100k.parquet
Log: data/logs/data_prep_20250206_143022.log

================================================================================
STEP 1: Initializing Spark Session
================================================================================
[OK] Spark session created

================================================================================
STEP 2: Loading 8 CIC-IDS2017 CSV Files
================================================================================
...
[OK] Loaded 2,830,743 rows from 88 columns

...

[SUCCESS] DATA PREPARATION COMPLETED
```

### Log File
- Detailed logs for each step
- Row counts at each stage
- Label distribution before/after sampling
- Verification results
- Any warnings or errors

## Troubleshooting

### "No files loaded!"
**Problem:** CSV files not found  
**Solution:** Check `MachineLearningCVE/` contains all 8 CSV files

### "Column 'label' not found"
**Problem:** Raw data has different column names  
**Solution:** Update `RAW_TO_INTERNAL_COLUMNS` in config.py

### Out of Memory
**Problem:** Spark runs out of memory  
**Solution:** Increase in `loader.py`:
```python
.config("spark.driver.memory", "8g")
.config("spark.executor.memory", "8g")
```

### Slow Sampling
**Problem:** Stratified sampling takes too long  
**Solution:** Normal (2-5 min), no optimization needed

## Advanced Usage

### Use in Other Scripts

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.loader import create_spark_session, load_all_cic_ids_files
from data.sampler import log_label_distribution

spark = create_spark_session("MyApp")
df = load_all_cic_ids_files(spark)
log_label_distribution(df)
```

### Use in Notebooks

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src")))

from data.text_builder import flow_to_text
from data.sampler import log_label_distribution
import pandas as pd

# Generate text for a single incident
text = flow_to_text(pd.Series({
    'label': 'DoS Hulk',
    'protocol': 6,
    'src_port': 52345,
    'dst_port': 80,
    'total_fwd_packets': 1000,
    # ... other fields
}))

print(text)
```

## Next Steps

1. ✅ Place raw CSV files
2. ✅ Run `python scripts/data/prepare_incident_data.py`
3. ✅ Run `python scripts/indexing/build_search_indexes.py`
4. ✅ Run `python scripts/search/interactive_hybrid_search.py`
5. ✅ Enjoy fast, semantic incident search!

## Documentation

- **Full Architecture:** See `DATA_PREP_ARCHITECTURE.md`
- **Implementation Details:** See `IMPLEMENTATION_SUMMARY.md`
- **Config Reference:** See `src/config.py` (well-commented)

## Performance

| Stage | Time | Data Size |
|-------|------|-----------|
| Load 8 CSVs | ~30 sec | 2.8M rows |
| Clean data | ~3-5 min | 2.3M rows |
| Deduplicate | ~1-2 min | 2.2M rows |
| Sample | ~30-60 sec | 100K rows |
| Convert to Pandas | ~10 sec | 100K rows |
| Generate text | ~20-30 sec | 100K rows |
| Save Parquet | ~5 sec | 30-50 MB |
| Verify | ~30 sec | ~100K rows |
| **Total** | **~5-10 min** | **Final: 30-50 MB** |

## Support

For issues:
1. Check console output and log file
2. Review config.py for settings
3. Verify raw CSV files in correct location
4. Check available disk space (~200 MB recommended)
5. Ensure Spark is properly installed

## What's Different from Original

| Feature | Before | After |
|---------|--------|-------|
| Architecture | Monolithic (1 file) | Modular (5 modules) |
| Reusability | Not reusable | Highly reusable |
| PySpark | Partial | Full pipeline |
| Memory Usage | 2.8M rows peak | 100K rows peak |
| Testing | Hard | Easy |
| Maintenance | Difficult | Simple |
| Logging | Minimal | Comprehensive |
| Documentation | None | Extensive |

**Key Benefit:** Same output, better code. Compatible with all existing search and RAG scripts.
