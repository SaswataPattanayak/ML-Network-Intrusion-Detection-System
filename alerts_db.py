"""
alerts_db.py
------------
Thread-safe SQLite wrapper shared by the inference engine and dashboard.

SQLite keeps the monitoring counters persistent while the browser's live
alert feed remains connection/session based. The dashboard deliberately does
not replay old rows on page refresh; it reads the aggregate counters from
SQLite and receives only newly generated alerts over WebSocket.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, List

from config import ALERTS_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    source_ip       TEXT,
    destination_ip  TEXT,
    src_port        INTEGER,
    dst_port        INTEGER,
    protocol        TEXT,
    attack_type     TEXT    NOT NULL,
    is_attack       INTEGER NOT NULL,
    confidence      REAL    NOT NULL,
    severity        TEXT,
    data_source     TEXT,
    raw_features    TEXT,
    class_probs     TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp);
"""

_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(ALERTS_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = get_connection()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


def insert_alert(payload: Dict[str, Any]) -> int:
    """Persist one structured inference alert and return its row id."""
    with _lock:
        conn = get_connection()
        try:
            cur = conn.execute(
                """
                INSERT INTO alerts
                    (timestamp, source_ip, destination_ip, src_port, dst_port,
                     protocol, attack_type, is_attack, confidence, severity,
                     data_source, raw_features, class_probs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["timestamp"],
                    payload.get("source_ip"),
                    payload.get("destination_ip"),
                    payload.get("src_port"),
                    payload.get("dst_port"),
                    payload.get("protocol"),
                    payload["attack_type"],
                    int(payload["is_attack"]),
                    payload["confidence"],
                    payload.get("severity"),
                    payload.get("data_source", "flow_record"),
                    json.dumps(payload.get("raw_features", {})),
                    json.dumps(payload.get("class_probabilities", {})),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def fetch_recent_alerts(limit: int = 100, attacks_only: bool = False) -> List[Dict[str, Any]]:
    with _lock:
        conn = get_connection()
        try:
            query = "SELECT * FROM alerts"
            if attacks_only:
                query += " WHERE is_attack = 1"
            query += " ORDER BY id DESC LIMIT ?"
            rows = conn.execute(query, (limit,)).fetchall()
            return [_row_to_dict(row) for row in rows]
        finally:
            conn.close()


def fetch_stats() -> Dict[str, Any]:
    """Return persistent monitoring counters, including severity totals."""
    with _lock:
        conn = get_connection()
        try:
            total = conn.execute(
                "SELECT COUNT(*) AS c FROM alerts"
            ).fetchone()["c"]
            attacks = conn.execute(
                "SELECT COUNT(*) AS c FROM alerts WHERE is_attack = 1"
            ).fetchone()["c"]

            by_type_rows = conn.execute(
                """
                SELECT attack_type, COUNT(*) AS c
                FROM alerts
                GROUP BY attack_type
                ORDER BY c DESC
                """
            ).fetchall()

            severity_rows = conn.execute(
                """
                SELECT severity, COUNT(*) AS c
                FROM alerts
                GROUP BY severity
                """
            ).fetchall()

            # Always return all four severity keys so the frontend never
            # falls back to zero simply because a severity has not appeared.
            by_severity = {
                "Critical": 0,
                "High": 0,
                "Medium": 0,
                "Low": 0,
            }
            for row in severity_rows:
                severity = row["severity"]
                if severity in by_severity:
                    by_severity[severity] = int(row["c"])

            return {
                "total_flows_seen": int(total),
                "total_attacks": int(attacks),
                "by_type": [
                    {"attack_type": row["attack_type"], "count": int(row["c"])}
                    for row in by_type_rows
                ],
                "by_severity": by_severity,
            }
        finally:
            conn.close()


def clear_alerts() -> int:
    """Delete all dynamic monitoring history and return the number removed."""
    with _lock:
        conn = get_connection()
        try:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM alerts"
            ).fetchone()["c"]
            conn.execute("DELETE FROM alerts")
            conn.commit()
            return int(count)
        finally:
            conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["is_attack"] = bool(d["is_attack"])

    try:
        d["raw_features"] = (
            json.loads(d["raw_features"])
            if d.get("raw_features")
            else {}
        )
    except (TypeError, json.JSONDecodeError):
        d["raw_features"] = {}

    try:
        d["class_probabilities"] = (
            json.loads(d.pop("class_probs"))
            if d.get("class_probs")
            else {}
        )
    except (TypeError, json.JSONDecodeError):
        d["class_probabilities"] = {}
        d.pop("class_probs", None)

    return d
