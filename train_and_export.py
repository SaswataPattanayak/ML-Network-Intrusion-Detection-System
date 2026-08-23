"""
train_and_export.py
--------------------
PHASE 1: Model & Preprocessing Extraction Script.

Reproduces IDS_Project4.ipynb's training pipeline exactly (same encoders,
same scaler, same 6 candidate models with the same hyperparameters, same
train-time noise injection) and exports everything the live system needs
as a single joblib bundle:

    model/model_bundle.joblib
        {
          "model":              <best-performing fitted sklearn estimator>,
          "scaler":              <fitted StandardScaler>,
          "protocol_encoder":    <fitted LabelEncoder for Protocol>,
          "label_encoder":       <fitted LabelEncoder for Label>,
          "feature_columns":     [...],
          "model_name":          "Random Forest" (e.g.),
          "trained_at":          ISO timestamp,
          "metadata":            {per-model accuracy, dataset shapes, ...}
        }

Run:
    python train_and_export.py \
        --train train_intrusion.csv \
        --test  test_intrusion.csv

This also regenerates the historical dashboard JSON (static/data/dashboard_data.json)
so Phase 4's dashboard keeps its "Model Health" / ROC / confusion-matrix sections
working exactly as before, from real training results.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config import (
    DASHBOARD_HISTORICAL_JSON,
    FEATURE_COLUMNS,
    MODEL_BUNDLE_PATH,
    RAW_TEST_CSV,
    RAW_TRAIN_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nids.train")


def round_float(value, digits=4):
    return round(float(value), digits)


# ----------------------------------------------------------------------
# Step 1 — preprocessing (mirrors notebook cell 2 exactly)
# ----------------------------------------------------------------------
def encode_dataset(train_data: pd.DataFrame, test_data: pd.DataFrame):
    train_df = train_data.copy()
    test_df = test_data.copy()

    drop_cols = ["SourceIP", "DestinationIP"]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    encoders = {}

    protocol_encoder = LabelEncoder()
    train_df["Protocol"] = protocol_encoder.fit_transform(train_df["Protocol"])
    test_df["Protocol"] = protocol_encoder.transform(test_df["Protocol"])
    encoders["Protocol"] = protocol_encoder

    label_encoder = LabelEncoder()
    train_df["Label"] = label_encoder.fit_transform(train_df["Label"])
    test_df["Label"] = label_encoder.transform(test_df["Label"])
    encoders["Label"] = label_encoder

    scaler = StandardScaler()
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove("Label")
    assert numerical_cols == FEATURE_COLUMNS, (
        f"Column order drifted from config.FEATURE_COLUMNS: {numerical_cols} != {FEATURE_COLUMNS}"
    )

    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

    return train_df, test_df, encoders, scaler


# ----------------------------------------------------------------------
# Step 2 — model zoo (identical hyperparameters to the notebook)
# ----------------------------------------------------------------------
def build_models():
    return {
        "Linear SVM": LinearSVC(random_state=42, max_iter=1500),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=21, weights="uniform"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=15, random_state=42,
        ),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=800, random_state=42),
    }


def train_all_models(X_train, y_train, X_test, y_test):
    models = build_models()

    # Notebook cell 4: noise is added ONCE, before any model is fit, and the
    # *same* noisy X_train/X_test are reused for every model in the loop.
    # Reproduced verbatim (including the fixed seed) so the exported model's
    # learned weights match what the notebook actually produced/evaluated.
    np.random.seed(42)
    X_train_noisy = X_train + np.random.normal(0, 0.03, X_train.shape)
    X_test_noisy = X_test + np.random.normal(0, 0.03, X_test.shape)

    results = []
    predictions = {}

    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_train_noisy, y_train)
        train_pred = model.predict(X_train_noisy)
        test_pred = model.predict(X_test_noisy)
        predictions[name] = test_pred

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        results.append({"name": name, "train_accuracy": round_float(train_acc),
                         "test_accuracy": round_float(test_acc)})
        logger.info("  -> test accuracy: %.4f", test_acc)

    results.sort(key=lambda item: item["test_accuracy"], reverse=True)
    best = results[0]
    return {
        "models": models,
        "results": results,
        "predictions": predictions,
        "best_name": best["name"],
        "best_model": models[best["name"]],
        "best_pred": predictions[best["name"]],
        "X_test_noisy": X_test_noisy,
    }


# ----------------------------------------------------------------------
# Step 3 — dashboard-analytics helpers (unchanged logic from prepare_dashboard.py,
# kept here so one script produces both the live-inference bundle AND the
# historical analytics JSON the dashboard already knows how to render).
# ----------------------------------------------------------------------
def sample_curve_points(fpr, tpr, limit=60):
    idx = list(range(len(fpr))) if len(fpr) <= limit else np.linspace(0, len(fpr) - 1, limit, dtype=int)
    sampled, last_pair = [], None
    for i in idx:
        pair = (round_float(fpr[i]), round_float(tpr[i]))
        if pair != last_pair:
            sampled.append({"fpr": pair[0], "tpr": pair[1]})
            last_pair = pair
    final_pair = {"fpr": round_float(fpr[-1]), "tpr": round_float(tpr[-1])}
    if not sampled or sampled[-1] != final_pair:
        sampled.append(final_pair)
    return sampled


def get_model_score_matrix(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores.reshape(-1, 1) if scores.ndim == 1 else scores
    raise ValueError(f"Model {model.__class__.__name__} does not support ROC scoring.")


def build_roc_curves(models, X_test, y_test, class_labels):
    y_test_bin = label_binarize(y_test, classes=np.arange(len(class_labels)))
    roc_models = []
    for name, model in models.items():
        score_matrix = get_model_score_matrix(model, X_test)
        class_curves, macro_aucs = [], []
        for idx, label in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(y_test_bin[:, idx], score_matrix[:, idx])
            class_auc = auc(fpr, tpr)
            macro_aucs.append(class_auc)
            class_curves.append({"label": str(label), "auc": round_float(class_auc),
                                  "points": sample_curve_points(fpr, tpr)})
        roc_models.append({"model": name, "macro_auc": round_float(np.mean(macro_aucs)),
                            "classes": class_curves})
    return roc_models


def class_distribution(series):
    counts = series.value_counts()
    total = int(counts.sum())
    return [{"label": str(l), "count": int(c), "share": round_float(c / total, 4)}
            for l, c in counts.items()]


def protocol_distribution(df):
    counts = df["Protocol"].value_counts()
    total = int(counts.sum())
    return [{"protocol": str(p), "count": int(c), "share": round_float(c / total, 4)}
            for p, c in counts.items()]


def convert_binary_labels(encoded_values, label_encoder):
    labels = label_encoder.inverse_transform(encoded_values)
    return np.where(labels == "Normal", "Normal", "Attack")


def build_model_confusion_matrices(y_test, predictions, label_encoder):
    y_test_binary = convert_binary_labels(y_test, label_encoder)
    labels = ["Normal", "Attack"]
    return [
        {"model": name, "labels": labels,
         "matrix": confusion_matrix(y_test_binary, convert_binary_labels(pred, label_encoder), labels=labels).tolist()}
        for name, pred in predictions.items()
    ]


def top_feature_importance(model, features):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        values = coef.mean(axis=0) if coef.ndim > 1 else coef
    else:
        return []
    pairs = sorted(zip(features, values), key=lambda item: float(item[1]), reverse=True)
    return [{"feature": f, "importance": round_float(v, 4)} for f, v in pairs[:6]]


def build_alert_samples(test_raw, actual_labels, predicted_labels):
    severity_map = {"Normal": "Low", "Probe": "Medium", "R2L": "High", "DoS": "High", "U2R": "Critical"}
    score_map = {"Normal": 18, "Probe": 58, "R2L": 76, "DoS": 85, "U2R": 96}

    alerts = test_raw.copy()
    alerts["ActualLabel"] = actual_labels
    alerts["PredictedLabel"] = predicted_labels
    alerts["Status"] = np.where(alerts["ActualLabel"] == alerts["PredictedLabel"], "Confirmed", "Needs Review")
    alerts["Severity"] = alerts["PredictedLabel"].map(severity_map).fillna("Medium")
    alerts["RiskScore"] = alerts["PredictedLabel"].map(score_map).fillna(50)
    alerts["ConnectionLoad"] = alerts["Connections"] + alerts["FailedLogins"] * 3

    class_order = ["DoS", "Normal", "Probe", "R2L", "U2R"]
    per_class_records = []
    for attack_class in class_order:
        class_slice = alerts[alerts["PredictedLabel"] == attack_class].sort_values(
            by=["Status", "ConnectionLoad", "BytesSent", "Duration"], ascending=[False, False, False, False],
        )
        per_class_records.append(class_slice.head(1))
    alert_view = pd.concat(per_class_records, ignore_index=True)

    records = []
    for _, row in alert_view.iterrows():
        records.append({
            "source_ip": row["SourceIP"], "destination_ip": row["DestinationIP"],
            "protocol": row["Protocol"], "duration": int(row["Duration"]),
            "bytes_sent": int(row["BytesSent"]), "bytes_received": int(row["BytesReceived"]),
            "failed_logins": int(row["FailedLogins"]), "connections": int(row["Connections"]),
            "predicted_label": row["PredictedLabel"], "actual_label": row["ActualLabel"],
            "status": row["Status"], "severity": row["Severity"], "risk_score": int(row["RiskScore"]),
        })
    return records


def build_dashboard_json(train_data, test_data, train_df, test_df, encoders, training_output):
    X_train = train_df.drop("Label", axis=1)
    X_test = test_df.drop("Label", axis=1)
    y_test = test_df["Label"]
    label_encoder = encoders["Label"]

    decoded_y_test = label_encoder.inverse_transform(y_test)
    decoded_best_pred = label_encoder.inverse_transform(training_output["best_pred"])

    report = classification_report(
        y_test, training_output["best_pred"], target_names=label_encoder.classes_,
        output_dict=True, zero_division=0,
    )
    best_result = next(r for r in training_output["results"] if r["name"] == training_output["best_name"])

    return {
        "meta": {"project": "Network Intrusion Detection System",
                 "generated_from": "IDS_Project4.ipynb", "generated_at": datetime.now(timezone.utc).isoformat()},
        "overview": {
            "train_rows": int(train_data.shape[0]), "test_rows": int(test_data.shape[0]),
            "feature_count": int(X_train.shape[1]),
            "attack_classes": [str(l) for l in label_encoder.classes_],
            "best_model": training_output["best_name"], "best_accuracy": best_result["test_accuracy"],
            "normal_share_test": round_float((test_data["Label"] == "Normal").mean()),
            "attack_share_test": round_float((test_data["Label"] != "Normal").mean()),
        },
        "model_results": training_output["results"],
        "class_distribution": {"train": class_distribution(train_data["Label"]),
                                "test": class_distribution(test_data["Label"])},
        "protocol_distribution": {"train": protocol_distribution(train_data),
                                   "test": protocol_distribution(test_data)},
        "feature_importance": top_feature_importance(training_output["best_model"], list(X_train.columns)),
        "model_confusion_matrices": build_model_confusion_matrices(y_test, training_output["predictions"], label_encoder),
        "roc_curves": build_roc_curves(training_output["models"], training_output["X_test_noisy"], y_test, label_encoder.classes_),
        "best_model_report": {
            "name": training_output["best_name"], "train_accuracy": best_result["train_accuracy"],
            "test_accuracy": best_result["test_accuracy"],
            "per_class": [{"label": l, "precision": round_float(m["precision"]), "recall": round_float(m["recall"]),
                           "f1_score": round_float(m["f1-score"]), "support": int(m["support"])}
                          for l, m in report.items() if l in label_encoder.classes_],
        },
        "alerts": build_alert_samples(test_data, decoded_y_test, decoded_best_pred),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train IDS models and export the live-inference bundle.")
    parser.add_argument("--train", type=str, default=str(RAW_TRAIN_CSV))
    parser.add_argument("--test", type=str, default=str(RAW_TEST_CSV))
    args = parser.parse_args()

    logger.info("Loading datasets ...")
    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)
    logger.info("Train shape: %s | Test shape: %s", train_data.shape, test_data.shape)

    train_df, test_df, encoders, scaler = encode_dataset(train_data, test_data)

    X_train = train_df.drop("Label", axis=1)
    y_train = train_df["Label"]
    X_test = test_df.drop("Label", axis=1)
    y_test = test_df["Label"]

    training_output = train_all_models(X_train, y_train, X_test, y_test)
    logger.info("Best model: %s (test accuracy %.4f)",
                training_output["best_name"],
                next(r["test_accuracy"] for r in training_output["results"] if r["name"] == training_output["best_name"]))

    # ---- Export the live-inference bundle (Phase 1 deliverable) ----
    bundle_payload = {
        "model": training_output["best_model"],
        "scaler": scaler,
        "protocol_encoder": encoders["Protocol"],
        "label_encoder": encoders["Label"],
        "feature_columns": FEATURE_COLUMNS,
        "model_name": training_output["best_name"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "all_model_results": training_output["results"],
            "train_rows": int(train_data.shape[0]),
            "test_rows": int(test_data.shape[0]),
        },
    }
    joblib.dump(bundle_payload, MODEL_BUNDLE_PATH)
    logger.info("Model bundle written to %s", MODEL_BUNDLE_PATH)

    # ---- Regenerate the historical dashboard JSON (Phase 4 data source) ----
    dashboard_json = build_dashboard_json(train_data, test_data, train_df, test_df, encoders, training_output)
    DASHBOARD_HISTORICAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_HISTORICAL_JSON.write_text(json.dumps(dashboard_json, indent=2), encoding="utf-8")
    logger.info("Dashboard analytics JSON written to %s", DASHBOARD_HISTORICAL_JSON)

    print("\n✅ Phase 1 complete.")
    print(f"   Best model : {training_output['best_name']}")
    print(f"   Bundle     : {MODEL_BUNDLE_PATH}")
    print(f"   Dashboard  : {DASHBOARD_HISTORICAL_JSON}")


if __name__ == "__main__":
    main()
