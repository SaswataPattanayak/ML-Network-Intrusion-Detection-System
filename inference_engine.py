"""
inference_engine.py
--------------------
PHASE 3: Asynchronous Flow Inference Engine.

Consumes flow records from a thread-safe Queue and runs them through the
model bundle
exported by Phase 1. This decoupling is the whole point: packet capture
never blocks on ML inference, and a slow inference batch never causes
dropped packets upstream.

For every classified flow:
  - a structured JSON alert payload is built (timestamp, source/destination
    IP, ports, protocol, attack type, confidence, per-class probabilities)
  - it's persisted to SQLite (alerts_db.py) for history/replay
  - it's appended to a line-delimited JSON log file (logs/alerts.jsonl) so
    it can be `tail -f`'d directly, per the "live-tailing log file" option
    in the brief
  - it's handed to an optional `on_alert` callback, which server.py (Phase 4)
    uses to broadcast over WebSocket without any file polling

Error handling: a single malformed/unparsable flow can never crash the
engine — it's logged and skipped, and the loop continues.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from config import ALERT_MIN_CONFIDENCE, ALERTS_JSONL_PATH, SEVERITY_MAP
from preprocessing import FlowPreprocessor, ModelBundle, PreprocessingError
import alerts_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nids.inference")

AlertCallback = Callable[[Dict], None]


def build_alert_payload(flow_dict: Dict, predicted_label: str, confidence: float,
                         class_probs: Dict[str, float]) -> Dict:
    """
    Structured JSON payload per the brief: timestamp, source IP, destination
    IP, attack type, confidence score — plus enough extra context (ports,
    protocol, severity, raw features) for the dashboard to render something
    genuinely useful rather than a bare classification.
    """
    is_attack = predicted_label != "Normal"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_ip": flow_dict.get("_source_ip", "unknown"),
        "destination_ip": flow_dict.get("_destination_ip", "unknown"),
        "src_port": flow_dict.get("_src_port"),
        "dst_port": flow_dict.get("_dst_port"),
        "protocol": flow_dict.get("Protocol"),
        "attack_type": predicted_label,
        "is_attack": is_attack,
        "confidence": round(confidence, 4),
        "severity": SEVERITY_MAP.get(predicted_label, "Medium"),
        "class_probabilities": {k: round(v, 4) for k, v in class_probs.items()},
        "data_source": "synthetic_demo" if flow_dict.get("_synthetic") else "flow_record",
        "raw_features": {
            "Duration": flow_dict.get("Duration"),
            "BytesSent": flow_dict.get("BytesSent"),
            "BytesReceived": flow_dict.get("BytesReceived"),
            "FailedLogins": flow_dict.get("FailedLogins"),
            "Connections": flow_dict.get("Connections"),
        },
    }


class InferenceEngine:
    """
    Runs on its own thread, pulling from `input_queue` (fed by Phase 2) and
    emitting alerts. Fully decoupled: construct once with a queue, call
    `start()`/`stop()`; everything else happens on the background thread.
    """

    def __init__(
        self,
        input_queue: "queue.Queue[Dict]",
        model_bundle: Optional[ModelBundle] = None,
        on_alert: Optional[AlertCallback] = None,
        persist_to_db: bool = True,
        write_jsonl: bool = True,
    ):
        self.input_queue = input_queue
        self.preprocessor = FlowPreprocessor(model_bundle)
        self.on_alert = on_alert
        self.persist_to_db = persist_to_db
        self.write_jsonl = write_jsonl

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.persist_to_db:
            alerts_db.init_db()

    def start(self) -> None:
        logger.info("Starting inference engine (model=%s) ...", self.preprocessor.bundle.model_name)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        logger.info("Stopping inference engine ...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                flow_dict = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_flow(flow_dict)
            except Exception:
                # Catch-all so one bad record can never kill the inference
                # thread — the loop must keep serving live traffic.
                logger.exception("Unexpected error processing flow; skipping.")

    def _process_flow(self, flow_dict: Dict) -> None:
        try:
            predicted_label, confidence, class_probs = self.preprocessor.predict(flow_dict)
        except PreprocessingError as exc:
            logger.warning("Skipping malformed flow: %s | flow=%s", exc, flow_dict)
            return

        if predicted_label != "Normal" and confidence < ALERT_MIN_CONFIDENCE:
            logger.debug("Suppressing low-confidence alert (%.2f < %.2f)", confidence, ALERT_MIN_CONFIDENCE)
            return

        payload = build_alert_payload(flow_dict, predicted_label, confidence, class_probs)
        self._emit(payload)

    def _emit(self, payload: Dict) -> None:
        if payload["is_attack"]:
            logger.info(
                "🚨 %s detected: %s -> %s (%s) confidence=%.2f",
                payload["attack_type"], payload["source_ip"], payload["destination_ip"],
                payload["protocol"], payload["confidence"],
            )

        if self.persist_to_db:
            try:
                alerts_db.insert_alert(payload)
            except Exception:
                logger.exception("Failed to persist alert to SQLite")

        if self.write_jsonl:
            try:
                with open(ALERTS_JSONL_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            except OSError:
                logger.exception("Failed to append to alerts.jsonl")

        if self.on_alert is not None:
            try:
                self.on_alert(payload)
            except Exception:
                logger.exception("on_alert callback raised an exception")


# ------------------------------------------------------------------------
# Standalone demo entrypoint: reads flows from stdin-fed synthetic queue
# for local testing without needing root/live sniffing. The real pipeline
# is wired together in run_live_nids.py.
# ------------------------------------------------------------------------
def _demo():
    import random

    q: "queue.Queue[Dict]" = queue.Queue()
    engine = InferenceEngine(q, on_alert=lambda p: None)
    engine.start()

    protocols = ["TCP", "UDP", "ICMP"]
    try:
        for _ in range(20):
            flow = {
                "Protocol": random.choice(protocols),
                "Duration": random.uniform(0, 40),
                "BytesSent": random.uniform(0, 3000),
                "BytesReceived": random.uniform(0, 1000),
                "FailedLogins": random.randint(0, 9),
                "Connections": random.uniform(0, 80),
                "_source_ip": f"10.0.0.{random.randint(2, 254)}",
                "_destination_ip": f"192.168.1.{random.randint(2, 254)}",
                "_src_port": random.randint(1024, 65535),
                "_dst_port": random.choice([22, 80, 443, 3389]),
            }
            q.put(flow)
            time.sleep(0.05)
        time.sleep(1.0)
    finally:
        engine.stop()


if __name__ == "__main__":
    _demo()
