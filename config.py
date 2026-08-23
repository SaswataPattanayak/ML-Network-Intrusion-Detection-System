"""
config.py
---------
Single source of truth for paths, feature schema, and tunable constants
shared across every phase (training, sniffing, inference, dashboard).

Keeping this centralized means the feature column order, model paths, and
network settings can never silently drift between the scripts that produce
data and the scripts that consume it.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Filesystem layout
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"

MODEL_BUNDLE_PATH = MODEL_DIR / "model_bundle.joblib"
ALERTS_DB_PATH = DATA_DIR / "alerts.db"
ALERTS_JSONL_PATH = LOG_DIR / "alerts.jsonl"          # human-tailable live log
DASHBOARD_HISTORICAL_JSON = STATIC_DIR / "data" / "dashboard_data.json"

for _dir in (MODEL_DIR, DATA_DIR, LOG_DIR, STATIC_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Feature schema — MUST match the column order the notebook fed into
# StandardScaler / the models: X = train_df.drop('Label', axis=1), where
# train_df's numeric columns (after dropping SourceIP/DestinationIP and
# label-encoding Protocol) are, in this exact order:
#   Protocol, Duration, BytesSent, BytesReceived, FailedLogins, Connections
# ----------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Protocol",
    "Duration",
    "BytesSent",
    "BytesReceived",
    "FailedLogins",
    "Connections",
]

RAW_TRAIN_CSV = BASE_DIR / "train_intrusion.csv"
RAW_TEST_CSV = BASE_DIR / "test_intrusion.csv"

# ----------------------------------------------------------------------
# Synthetic dashboard demonstration
# ----------------------------------------------------------------------
SYNTHETIC_FLOW_INTERVAL_SECONDS = 1.5
SYNTHETIC_SOURCE_NETWORKS = [
    ("10.10.0", "192.168.10"),
    ("10.20.0", "192.168.20"),
    ("172.16.10", "172.16.20"),
]

# ----------------------------------------------------------------------
# Inference / alerting (Phase 3) settings
# ----------------------------------------------------------------------
INFERENCE_QUEUE_MAXSIZE = 5000

# Synthetic dashboard demonstration mode (no packet capture / Npcap required).
SYNTHETIC_FLOW_INTERVAL_SECONDS = 1.5
SYNTHETIC_SOURCE_NETWORKS = [("10.10.0", "192.168.10"), ("10.20.0", "192.168.20"), ("172.16.10", "172.16.20")]     # backpressure guard between sniffer and inference threads
ALERT_MIN_CONFIDENCE = 0.0         # set >0 (e.g. 0.6) to suppress low-confidence attack alerts

SEVERITY_MAP = {
    "Normal": "Low",
    "Probe": "Medium",
    "DoS": "High",
    "R2L": "High",
    "U2R": "Critical",
}

# ----------------------------------------------------------------------
# Dashboard server (Phase 4) settings
# ----------------------------------------------------------------------
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
WS_BROADCAST_QUEUE_MAXSIZE = 1000
