---
title: "AI-Powered Cybersecurity Incident Response Assistant"
subtitle: "Dataset Details"
tagline: "CIC-IDS2017: source, format, key fields, purpose and the preprocessing performed for retrieval and machine learning"
project: "Capstone Project 8 · CIC-IDS2017"
date: "October 2026"
---

# 1. Dataset summary

| Property | Value |
|---|---|
| **Dataset name** | CIC-IDS2017 – Intrusion Detection Evaluation Dataset |
| **Publisher** | Canadian Institute for Cybersecurity (CIC), University of New Brunswick |
| **Primary link** | https://www.unb.ca/cic/datasets/ids-2017.html |
| **Alternative source** | Kaggle mirror of CIC-IDS2017 (clean Parquet version), as referenced in the project brief |
| **Reference** | I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018 |
| **Release used** | `MachineLearningCVE`: 8 CSV files of labelled network flows (CICFlowMeter features) |
| **Format** | CSV (one row per bidirectional network flow); the dataset is also published as PCAP |
| **Size** | 2,830,743 flows · 79 columns · 15 traffic labels (BENIGN + 14 attack types) |
| **Capture** | Five working days (Monday–Friday, July 2017) of a realistic network with benign user traffic and scheduled attacks |

# 2. Source files

| File | Day and content | Flows |
|---|---|---:|
| `Monday-WorkingHours.pcap_ISCX.csv` | Monday: benign traffic only | 529,918 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | Tuesday: FTP-Patator, SSH-Patator | 445,909 |
| `Wednesday-workingHours.pcap_ISCX.csv` | Wednesday: DoS (slowloris, Slowhttptest, Hulk, GoldenEye), Heartbleed | 692,703 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Thursday morning: web brute force, XSS, SQL injection | 170,366 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Thursday afternoon: infiltration | 288,602 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Friday morning: botnet (Bot) | 191,033 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday afternoon: port scan | 286,467 |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Friday afternoon: DDoS | 225,745 |
| **Total** | | **2,830,743** |

The files are placed in `MachineLearningCVE/` (not in git). A small committed sample is in
`data/sample_data/` (see §7).

# 3. Key fields

## 3.1 Fields named in the brief and their availability

The brief lists the key fields of the full CIC-IDS2017 release. The `MachineLearningCVE` release
used here contains the flow statistics and labels but **not** the identity fields:

| Field | In this release | Used for |
|---|---|---|
| Destination Port | ✅ | Metadata filter, incident text, prioritisation context (not a model feature) |
| Flow Duration, Total Fwd / Backward Packets, Total Length of Fwd / Bwd Packets | ✅ | Model features, incident text, numeric search filters |
| Flow Bytes/s, Flow Packets/s, Flow IAT Mean | ✅ | Flow Packets/s is a model feature and a search filter; the others are in the raw table |
| Packet Length Mean, Average Packet Size | ✅ | Model features (Packet Length Mean) |
| SYN / RST / ACK Flag Count | ✅ | Model features |
| Active Mean, Idle Mean | ✅ | Model features |
| Label | ✅ | Ground truth for classification, evaluation and search filters |
| Flow ID, Source IP, Source Port, Destination IP, Protocol, Timestamp | ❌ not in this release | Filters implemented, but have no data (see §6.3) |

## 3.2 The 24 features used by the models and the assistant

| Feature (internal name) | CIC-IDS2017 column | Meaning |
|---|---|---|
| `flow_duration` | Flow Duration | Total flow duration, microseconds |
| `total_fwd_packets` | Total Fwd Packets | Packets from source to destination |
| `total_bwd_packets` | Total Bwd Packets | Packets from destination back to source |
| `total_len_fwd` | Total Length of Fwd Packets | Bytes sent forward |
| `total_len_bwd` | Total Length of Bwd Packets | Bytes sent backward |
| `flow_packets_s` | Flow Packets/s | Packets per second over the flow |
| `fwd_packets_s` | Fwd Packets/s | Forward packets per second |
| `bwd_packets_s` | Bwd Packets/s | Backward packets per second |
| `fwd_packet_length_mean` | Fwd Packet Length Mean | Mean forward packet size, bytes |
| `bwd_packet_length_mean` | Bwd Packet Length Mean | Mean backward packet size, bytes |
| `min_packet_length` | Min Packet Length | Smallest packet, bytes |
| `max_packet_length` | Max Packet Length | Largest packet, bytes |
| `packet_length_mean` | Packet Length Mean | Mean packet size (both directions), bytes |
| `packet_length_std` | Packet Length Std | Standard deviation of packet size, bytes |
| `syn_flag_count` | SYN Flag Count | 1 if a TCP SYN flag was seen |
| `ack_flag_count` | ACK Flag Count | 1 if a TCP ACK flag was seen |
| `rst_flag_count` | RST Flag Count | 1 if a TCP RST flag was seen |
| `psh_flag_count` | PSH Flag Count | 1 if a TCP PSH flag was seen |
| `fin_flag_count` | FIN Flag Count | 1 if a TCP FIN flag was seen |
| `down_up_ratio` | Down/Up Ratio | Backward ÷ forward packets |
| `init_win_bytes_forward` | Init_Win_bytes_forward | Initial TCP window, forward, bytes (−1 = not set) |
| `init_win_bytes_backward` | Init_Win_bytes_backward | Initial TCP window, backward, bytes (−1 = not set) |
| `active_mean` | Active Mean | Mean active time before idle, microseconds |
| `idle_mean` | Idle Mean | Mean idle time before active, microseconds |

## 3.3 Labels

| Label | Category | Flows after deduplication | In the incident library |
|---|---|---:|---:|
| BENIGN | Normal traffic | 1,820,651 | 52,210 |
| DoS Hulk | Denial of service | 166,335 | 7,819 |
| DDoS | Distributed denial of service | 127,188 | 7,843 |
| PortScan | Reconnaissance | 90,743 | 7,828 |
| DoS GoldenEye | Denial of service | 10,281 | 5,246 |
| DoS Slowhttptest | Denial of service | 5,172 | 2,584 |
| DoS slowloris | Denial of service | 4,867 | 2,634 |
| FTP-Patator | Brute force (FTP) | 4,259 | 4,259 |
| SSH-Patator | Brute force (SSH) | 3,149 | 3,149 |
| Bot | Botnet | 1,936 | 1,936 |
| Web Attack - Brute Force | Web attack | 1,426 | 1,323 |
| Web Attack - XSS | Web attack | 651 | 651 |
| Infiltration | Infiltration | 36 | 36 |
| Web Attack - Sql Injection | Web attack | 20 | 20 |
| Heartbleed | Exploit | 11 | 11 |
| **Total** | | **2,236,725** | **97,549** |

# 4. Purpose in this project

| Use | How the dataset supports it |
|---|---|
| **Historical incident similarity (RAG)** | Each sampled flow becomes an incident with an ID and a text description; analysts search them with hybrid retrieval |
| **Hybrid search and metadata filtering** | Attack label and destination port as metadata; flow statistics as numeric filters ("more than 10000 packets/s") |
| **Attack-type classification** | 15 labelled classes for a supervised classifier |
| **Intrusion-risk prediction** | BENIGN vs attack labels for a calibrated risk model |
| **Anomaly detection** | BENIGN flows only, to learn what normal traffic looks like |
| **Explainability** | Flow statistics with clear meanings, explained with SHAP and compared with normal-traffic medians |
| **Prioritisation** | Model outputs plus attack-type severity → P1–P4 |
| **Evaluation** | Historical labels are the ground truth for the evaluation cases and the exact checks |

# 5. Preprocessing performed

## 5.1 Incident library (PySpark, `scripts/data/prepare_incident_data.py`)

| Step | What happens | Rows |
|---|---|---:|
| Load | 8 CSVs read with PySpark and aligned to one schema (79 raw columns → internal names) | 2,830,743 |
| Normalise columns | Headers with stray spaces and variants (`Destination Port` / `Dst Port`) mapped to snake_case names | |
| Cast | 40 numeric columns cast to double | |
| Infinity → null | `±inf` values (common in Flow Bytes/s and Flow Packets/s) set to null | |
| Standardise labels | Labels trimmed; the broken Web Attack separator (`Web Attack � Brute Force`) repaired; 15 canonical names | |
| Deduplicate | SHA-256 hash over 11 flow-identity fields; duplicate flows dropped | 2,236,725 |
| Drop incomplete | Rows without a label or destination port removed (none in this release) | 2,236,725 |
| Stratified sample | Per-label targets (BENIGN 52,000, large attack classes about 7,800, smaller ones kept whole), seed 42 | 97,549 |
| IDs and text | Shuffled, given IDs `INC-000001`…`INC-097549`, and a template-based `incident_text` (no LLM; missing values omitted) | 97,549 |
| Validate | Required columns, unique IDs, non-empty texts, no duplicates, label distribution | |
| Output | `data/processed/incidents_100k.parquet` (41 columns) | |
| Index | BM25 keyword index and all-MiniLM-L6-v2 embeddings in ChromaDB, aligned and verified against the Parquet | |

Example incident text (one flow = one document, 241 tokens at most, so it is never split):

> *Attack: DoS slowloris. Destination port: 80. Flow duration: 3001067 microseconds. Forward
> packets: 3. Backward packets: 0. Forward bytes: 0. Backward bytes: 0. Flow packets per second:
> 0.999644. …*

## 5.2 Training table for the models (`ml/eda/data_cleaning.ipynb`)

Starting from the incident library:

1. Dropped the all-null identity columns (IPs, source port, protocol, timestamp) and constant
   columns.
2. Dropped redundant per-direction flag columns.
3. Dropped rows with a negative flow duration (1 row).
4. Filled the few missing `flow_packets_s` values with the median.
5. Removed `incident_id` and `incident_text`.

Result: `data/cleaned_network_data/cleaned_network_data.csv`, with 97,548 rows: label, `dst_port`
and the 24 features.

Further steps in the training notebooks:

- **`dst_port` is excluded** from the model features, so the models learn traffic behaviour rather
  than "port X means attack Y".
- **Duplicate feature vectors are removed before** the train/test split, so no flow appears in both.
- Splits are stratified, with seed 42 (the risk model also keeps a separate split for probability calibration).
- The anomaly detector uses an arcsinh transform and is trained on BENIGN flows only.

# 6. Data quality notes and limitations

## 6.1 Class imbalance

BENIGN is 81% of the deduplicated flows, and Heartbleed (11), SQL injection (20) and Infiltration
(36) are extremely rare. Sampling keeps every rare flow and caps the large classes, and the
classifier uses balanced class weights. Macro F1 (0.839) is reported next to accuracy (0.986)
because the rare classes are hardest.

## 6.2 Duplicates

594,018 duplicate flows (21%) were removed. Without this step, identical flows could appear in
both the training and test sets and inflate accuracy.

## 6.3 Missing identity fields

This release has no IP, protocol, source port or timestamp values. The metadata filters for these
fields are implemented and validated against the fields actually populated. Asking for them
returns a clear message instead of an empty result. The `GeneratedLabelledFlows` release of
CIC-IDS2017 includes them and could be used to enable these filters.

## 6.4 Single-flow incidents

Each incident is one flow. Behaviour across many flows (many sources to one target, slow scans
over hours) is not visible in a single record. The assistant states this in its limitations.

## 6.5 Dataset artifacts

The risk model relies partly on `init_win_bytes_forward`, which is known to separate classes
partly because of how the dataset was generated. SHAP makes this visible; it is documented as a
limitation.

# 7. Sample data in the repository

`data/sample_data/` (committed):

| File | Contents |
|---|---|
| `sample_incidents.csv` | 120 incidents, the first 8 per label: ID, label, port, 24 features, incident text |
| `triage_queue_sample.csv` | 14 rows (incident IDs, metric rows, a partial flow) for the Triage Queue and `POST /ml/prioritize/csv` |
| `sample_queries.json` | 8 assistant questions with expected facts and real answer excerpts |

Regenerate them with `python scripts/data/make_sample_data.py`.
