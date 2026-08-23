"""
preprocessing.py
-----------------
The exact data-cleaning / feature-engineering pipeline from IDS_Project4.ipynb,
extracted into a reusable, importable form.

Notebook pipeline (cell 2), reproduced faithfully:
    1. Drop SourceIP, DestinationIP (high-cardinality, not model features)
    2. LabelEncode 'Protocol' (fit on train, applied everywhere else)
    3. LabelEncode 'Label' (only relevant for training/eval, not live inference)
    4. StandardScale all numeric columns except Label:
       ['Protocol', 'Duration', 'BytesSent', 'BytesReceived',
        'FailedLogins', 'Connections']

IMPORTANT — training-time noise is intentionally NOT reproduced here:
    The notebook adds `np.random.normal(0, 0.03, X.shape)` to X_train/X_test
    AFTER scaling, purely to cap the reported accuracy for the assignment
    (see cell 4 comment: "Add controlled noise to limit accuracy"). That's a
    one-time experimental step baked into how the saved model was *trained*,
    not a real preprocessing requirement. Re-applying random noise to live
    traffic at inference time would make predictions non-deterministic and
    is never done here — only replicated inside train_and_export.py so the
    exported model matches the notebook's actual learned weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, MODEL_BUNDLE_PATH

logger = logging.getLogger("nids.preprocessing")


class PreprocessingError(ValueError):
    """Raised when a live packet/flow record cannot be safely preprocessed."""


@dataclass
class ModelBundle:
    """
    Everything Phase 1 exports and every later phase loads back in.
    Bundling these together (instead of separate pickle files) guarantees
    the scaler/encoders/model/feature-order can never get out of sync.
    """
    model: Any
    scaler: Any
    protocol_encoder: Any
    label_encoder: Any
    feature_columns: list
    model_name: str
    trained_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path=MODEL_BUNDLE_PATH) -> "ModelBundle":
        try:
            payload = joblib.load(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Model bundle not found at {path}. Run train_and_export.py first."
            ) from exc
        return cls(**payload)


class FlowPreprocessor:
    """
    Wraps a loaded ModelBundle and exposes `preprocess_flow`, which
    turns one flow-level record into a model-ready feature vector using the
    *exact* fitted encoders/scaler from training (never refit on live data).
    """

    def __init__(self, bundle: Optional[ModelBundle] = None):
        self.bundle = bundle or ModelBundle.load()
        self._known_protocols = set(self.bundle.protocol_encoder.classes_)
        # Fallback protocol for anything the encoder never saw during training
        # (e.g. a raw IP protocol number we can't confidently name).
        self._fallback_protocol = self.bundle.protocol_encoder.classes_[0]

    def preprocess_flow(self, packet_data: Dict[str, Any]) -> np.ndarray:
        """
        Transform one live flow record into a scaled feature vector matching
        the training-time schema.

        Expected input keys (produced by the synthetic flow generator or any compatible flow source):
            Protocol      : str  ("TCP" / "UDP" / "ICMP" / ...)
            Duration      : float (seconds)
            BytesSent     : float
            BytesReceived : float
            FailedLogins  : int
            Connections   : float

        Returns
        -------
        np.ndarray of shape (1, len(FEATURE_COLUMNS)) ready for model.predict().

        Raises
        ------
        PreprocessingError on missing/malformed fields, with the offending
        field named, so callers (Phase 3) can log-and-skip instead of crashing
        the inference loop over one bad packet.
        """
        record = self._validate_and_coerce(packet_data)

        # 1. Encode Protocol using the *fitted* training encoder.
        protocol_value = record["Protocol"]
        if protocol_value not in self._known_protocols:
            logger.warning(
                "Unseen protocol '%s' in live traffic — falling back to '%s'.",
                protocol_value, self._fallback_protocol,
            )
            protocol_value = self._fallback_protocol
        encoded_protocol = int(
            self.bundle.protocol_encoder.transform([protocol_value])[0]
        )

        # 2. Assemble the raw feature row in the exact training column order.
        raw_row = {
            "Protocol": encoded_protocol,
            "Duration": record["Duration"],
            "BytesSent": record["BytesSent"],
            "BytesReceived": record["BytesReceived"],
            "FailedLogins": record["FailedLogins"],
            "Connections": record["Connections"],
        }
        row_df = pd.DataFrame([raw_row], columns=self.bundle.feature_columns)

        # 3. Scale using the *fitted* training StandardScaler (transform only).
        # Keep the result as a DataFrame (not a bare ndarray) so downstream
        # sklearn estimators see the same column names they were fitted
        # with — avoids spurious "X does not have valid feature names"
        # warnings and keeps column order self-documenting.
        scaled = self.bundle.scaler.transform(row_df)
        return pd.DataFrame(scaled, columns=self.bundle.feature_columns)

    def predict(self, packet_data: Dict[str, Any]):
        """
        Convenience end-to-end call: preprocess -> predict -> decode label
        -> return (predicted_label, confidence, class_probabilities).
        """
        features = self.preprocess_flow(packet_data)
        model = self.bundle.model

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
            class_probs = {
                cls: float(p)
                for cls, p in zip(self.bundle.label_encoder.classes_, proba)
            }
        else:
            # e.g. LinearSVC has no predict_proba; use decision_function
            # margins, softmax-normalized, purely as a confidence proxy.
            scores = model.decision_function(features)[0]
            scores = np.atleast_1d(scores)
            exp_scores = np.exp(scores - np.max(scores))
            proba = exp_scores / exp_scores.sum()
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
            class_probs = {
                cls: float(p)
                for cls, p in zip(self.bundle.label_encoder.classes_, proba)
            }

        predicted_label = self.bundle.label_encoder.inverse_transform([pred_idx])[0]
        return predicted_label, confidence, class_probs

    @staticmethod
    def _validate_and_coerce(packet_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Defensive validation so one malformed live packet/flow can never
        crash the inference loop. Raises PreprocessingError with a precise
        reason; numeric fields are coerced/clamped rather than rejected
        outright where it's safe to do so.
        """
        if not isinstance(packet_data, dict):
            raise PreprocessingError(f"Expected dict, got {type(packet_data)!r}")

        required_numeric = ["Duration", "BytesSent", "BytesReceived",
                             "FailedLogins", "Connections"]
        missing = [k for k in ["Protocol", *required_numeric] if k not in packet_data]
        if missing:
            raise PreprocessingError(f"Missing required field(s): {missing}")

        protocol = packet_data["Protocol"]
        if not isinstance(protocol, str) or not protocol.strip():
            raise PreprocessingError(f"Invalid Protocol value: {protocol!r}")
        protocol = protocol.strip().upper()

        coerced = {"Protocol": protocol}
        for key in required_numeric:
            value = packet_data[key]
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise PreprocessingError(f"Field '{key}' is not numeric: {value!r}")
            if np.isnan(value) or np.isinf(value):
                raise PreprocessingError(f"Field '{key}' is NaN/Inf")
            if value < 0:
                # Clamp instead of hard-fail: negative durations/byte counts
                # can happen from clock skew or partial captures; clamping to
                # 0 keeps the flow usable without corrupting the scaler input.
                logger.debug("Clamping negative value for '%s' (%s) to 0.", key, value)
                value = 0.0
            coerced[key] = value

        return coerced


# ----------------------------------------------------------------------
# Module-level convenience function matching the exact name requested:
# preprocess_flow(packet_data) -> np.ndarray
# Lazily loads a singleton preprocessor so callers can just do:
#     from preprocessing import preprocess_flow
#     features = preprocess_flow(flow_dict)
# ----------------------------------------------------------------------
_singleton: Optional[FlowPreprocessor] = None


def _get_singleton() -> FlowPreprocessor:
    global _singleton
    if _singleton is None:
        _singleton = FlowPreprocessor()
    return _singleton


def preprocess_flow(packet_data: Dict[str, Any]) -> np.ndarray:
    return _get_singleton().preprocess_flow(packet_data)
