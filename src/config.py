"""
Configuration and constants for the cybersecurity incident response system.
Paths, raw filenames, column mappings, and per-class sampling limits.
"""
import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "MachineLearningCVE"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Create processed data directory if it doesn't exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Raw CSV filenames (8 official CIC-IDS2017 files)
# ============================================================================
RAW_CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# ============================================================================
# Raw column names (with leading spaces) → internal column names
# ============================================================================
# ML feature columns (16 total) + context columns (6) + label (1)
RAW_TO_INTERNAL_COLUMNS = {
    " Protocol": "protocol",
    " Flow Duration": "flow_duration",
    " Destination Port": "dst_port",
    " Dst Port": "dst_port",
    " Total Fwd Packets": "total_fwd_packets",
    " Total Bwd Packets": "total_bwd_packets",
    " Total Backward Packets": "total_bwd_packets",
    " Total Length of Fwd Packets": "total_len_fwd",
    " Total Length of Bwd Packets": "total_len_bwd",
    " Fwd Packet Length Mean": "fwd_packet_length_mean",
    " Bwd Packet Length Mean": "bwd_packet_length_mean",
    " Flow Bytes/s": "flow_bytes_s",
    " Flow Packets/s": "flow_packets_s",
    " Fwd Packets/s": "fwd_packets_s",
    " Bwd Packets/s": "bwd_packets_s",
    " Flow IAT Mean": "flow_iat_mean",
    " Min Packet Length": "min_packet_length",
    " Max Packet Length": "max_packet_length",
    " Packet Length Mean": "packet_length_mean",
    " Packet Length Std": "packet_length_std",
    " Fwd PSH Flags": "fwd_psh_flags",
    " Bwd PSH Flags": "bwd_psh_flags",
    " Fwd URG Flags": "fwd_urg_flags",
    " Bwd URG Flags": "bwd_urg_flags",
    " FIN Flag Count": "fin_flag_count",
    " SYN Flag Count": "syn_flag_count",
    " RST Flag Count": "rst_flag_count",
    " PSH Flag Count": "psh_flag_count",
    " ACK Flag Count": "ack_flag_count",
    " URG Flag Count": "urg_flag_count",
    " Fwd RST Flags": "fwd_rst_flags",
    " Bwd RST Flags": "bwd_rst_flags",
    " Fwd SYN Flags": "fwd_syn_flags",
    " Bwd SYN Flags": "bwd_syn_flags",
    " Fwd ACK Flags": "fwd_ack_flags",
    " Bwd ACK Flags": "bwd_ack_flags",
    " Down/Up Ratio": "down_up_ratio",
    " Init_Win_bytes_forward": "init_win_bytes_forward",
    " Init_Win_bytes_backward": "init_win_bytes_backward",
    " Active Mean": "active_mean",
    " Idle Mean": "idle_mean",
    " Src IP": "src_ip",
    " Src Port": "src_port",
    " Dst IP": "dst_ip",
    " Timestamp": "timestamp",
    " Label": "label",
}

# ============================================================================
# ML feature columns (16 total - excluding IPs, ports, timestamp)
# ============================================================================
ML_FEATURE_COLUMNS = [
    "protocol",
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_len_fwd",
    "total_len_bwd",
    "fwd_packet_length_mean",
    "bwd_packet_length_mean",
    "flow_bytes_s",
    "flow_packets_s",
    "fwd_packets_s",
    "bwd_packets_s",
    "flow_iat_mean",
    "min_packet_length",
    "max_packet_length",
    "packet_length_mean",
    "packet_length_std",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "down_up_ratio",
    "init_win_bytes_forward",
    "init_win_bytes_backward",
    "active_mean",
    "idle_mean",
    "fwd_psh_flags",
    "bwd_psh_flags",
    "fwd_urg_flags",
    "bwd_urg_flags",
    "fwd_rst_flags",
    "bwd_rst_flags",
    "fwd_syn_flags",
    "bwd_syn_flags",
    "fwd_ack_flags",
    "bwd_ack_flags",
]

# ============================================================================
# Context columns (kept for reference/display, not used as ML features)
# ============================================================================
CONTEXT_COLUMNS = [
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "timestamp",
]

# Label column
LABEL_COLUMN = "label"

# Features used to describe incidents for retrieval. Missing source fields are
# omitted when preparing the dataset rather than synthesized.
RETRIEVAL_TEXT_COLUMNS = [
    LABEL_COLUMN,
    "protocol",
    "src_port",
    "dst_port",
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_len_fwd",
    "total_len_bwd",
    "flow_packets_s",
    "fwd_packets_s",
    "bwd_packets_s",
    "fwd_packet_length_mean",
    "bwd_packet_length_mean",
    "min_packet_length",
    "max_packet_length",
    "packet_length_mean",
    "packet_length_std",
    "syn_flag_count",
    "ack_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "fin_flag_count",
    "fwd_syn_flags",
    "bwd_syn_flags",
    "fwd_ack_flags",
    "bwd_ack_flags",
    "fwd_rst_flags",
    "bwd_rst_flags",
    "fwd_psh_flags",
    "bwd_psh_flags",
    "down_up_ratio",
    "init_win_bytes_forward",
    "init_win_bytes_backward",
    "active_mean",
    "idle_mean",
]

# ============================================================================
# Label standardization map (handle inconsistent spelling/casing)
# ============================================================================
LABEL_STANDARDIZATION = {
    "BENIGN": "BENIGN",
    "FTP-Patator": "FTP-Patator",
    "SSH-Patator": "SSH-Patator",
    "DoS Hulk": "DoS Hulk",
    "DoS GoldenEye": "DoS GoldenEye",
    "DoS slowloris": "DoS slowloris",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "Heartbleed": "Heartbleed",
    "Infiltration": "Infiltration",
    "Web Attack - Brute Force": "Web Attack - Brute Force",
    "Web Attack - XSS": "Web Attack - XSS",
    "Web Attack - Sql Injection": "Web Attack - Sql Injection",
    "Bot": "Bot",
    "PortScan": "PortScan",
    "DDoS": "DDoS",
}

# ============================================================================
# Stratified sampling targets (approximately 100K total)
# ============================================================================
SAMPLES_PER_CLASS_100K = {
    "BENIGN": 52000,
    "DoS Hulk": 7800,
    "PortScan": 7800,
    "DDoS": 7800,
    "DoS GoldenEye": 5200,
    "FTP-Patator": 5200,
    "SSH-Patator": 5200,
    "DoS slowloris": 2600,
    "DoS Slowhttptest": 2600,
    "Bot": 2600,
    "Web Attack - Brute Force": 1300,
    "Web Attack - XSS": 1300,
    "Web Attack - Sql Injection": 520,
    "Infiltration": None,  # Keep all
    "Heartbleed": None,    # Keep all
}

# ============================================================================
# Random seed for reproducibility
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# Output filenames
# ============================================================================
CLEANED_INCIDENTS_FILE = DATA_PROCESSED_DIR / "incidents_100k.parquet"

# ============================================================================
# Embedding and search settings
# ============================================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "incidents"
RRF_K = 60  # Constant for Reciprocal Rank Fusion formula
BM25_WEIGHT = 1.0
VECTOR_WEIGHT = 1.0
