"""Train, compare, and calibrate the lead-scoring model. Run manually, not from the app."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from clean_data import TARGET_COLUMN, DataCleaner
from features import BinaryMapper, build_preprocessor, make_preprocessor

RANDOM_STATE = 42
DATA_PATH = Path("data/Lead_Scoring.csv")
MODEL_DIR = Path("model")
PRECISION_TARGET = 0.80
CALIBRATION_GAP_THRESHOLD = 0.10  # reliability-curve gap beyond this counts as "poorly calibrated"


def get_candidates() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1
        ),
    }


def make_full_pipeline(numeric_cols, binary_cols, categorical_cols, estimator) -> Pipeline:
    """Clean -> binarize -> encode -> model, as one pipeline. Every step
    refits from scratch whenever this pipeline is cloned and fit, which is
    what makes cross-validation and the final scoring artifact leakage-safe."""
    preprocessor = make_preprocessor(numeric_cols, binary_cols, categorical_cols)
    return Pipeline(
        [
            ("clean", DataCleaner()),
            ("binarize", BinaryMapper()),
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def cross_validate_candidates(X_train, y_train, numeric_cols, binary_cols, categorical_cols, candidates) -> list[dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for name, estimator in candidates.items():
        pipe = make_full_pipeline(numeric_cols, binary_cols, categorical_cols, estimator)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        results.append(
            {"model": name, "roc_auc_mean": float(scores.mean()), "roc_auc_std": float(scores.std())}
        )
    return results


def check_calibration(y_true, y_prob, n_bins=10) -> tuple[bool, float, float]:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    max_gap = float(np.max(np.abs(prob_true - prob_pred)))
    brier = float(brier_score_loss(y_true, y_prob))
    poorly_calibrated = max_gap > CALIBRATION_GAP_THRESHOLD
    return poorly_calibrated, max_gap, brier


def find_threshold(y_true, score, target_precision=PRECISION_TARGET) -> tuple[int, float, float, bool]:
    """Lowest score threshold on the test-set precision-recall curve with precision >= target."""
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    precision, recall = precision[:-1], recall[:-1]

    candidates = [
        (int(t), p, r) for t, p, r in zip(thresholds, precision, recall) if p >= target_precision
    ]
    if not candidates:
        best_idx = int(np.argmax(precision))
        return int(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx]), False

    candidates.sort(key=lambda c: c[0])
    t, p, r = candidates[0]
    return t, float(p), float(r), True


def extract_feature_importance(fitted_pipe: Pipeline) -> list[dict]:
    model = fitted_pipe.named_steps["model"]
    feature_names = list(fitted_pipe.named_steps["preprocess"].get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return []

    ranked = sorted(zip(feature_names, values), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": name, "importance": float(val)} for name, val in ranked]


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    X = raw.drop(columns=[TARGET_COLUMN])
    y = raw[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # decide column structure once, from the training fold only, then bake
    # it into every pipeline built below (this is a structural choice, not
    # a statistic that can leak target information the way an imputed
    # median or a rare-category cutoff can)
    probe_prepared = BinaryMapper().fit_transform(DataCleaner().fit_transform(X_train))
    numeric_cols, binary_cols, categorical_cols = build_preprocessor(probe_prepared)[1:]
    feature_columns = numeric_cols + binary_cols + categorical_cols

    candidates = get_candidates()
    cv_results = cross_validate_candidates(
        X_train, y_train, numeric_cols, binary_cols, categorical_cols, candidates
    )
    print("Cross-validated ROC-AUC:")
    for r in cv_results:
        print(f"  {r['model']}: {r['roc_auc_mean']:.4f} (+/- {r['roc_auc_std']:.4f})")

    best_name = max(cv_results, key=lambda r: r["roc_auc_mean"])["model"]
    print(f"Best candidate: {best_name}")

    best_pipe = make_full_pipeline(numeric_cols, binary_cols, categorical_cols, candidates[best_name])
    best_pipe.fit(X_train, y_train)

    # feature importance is read off this uncalibrated fit -- calibration
    # only rescales probabilities, it doesn't change which features drove it
    feature_importance = extract_feature_importance(best_pipe)

    y_prob_raw = best_pipe.predict_proba(X_test)[:, 1]
    test_roc_auc = float(roc_auc_score(y_test, y_prob_raw))
    poorly_calibrated, max_gap, brier_before = check_calibration(y_test, y_prob_raw)

    calibration_applied = False
    y_prob_final = y_prob_raw
    brier_after = None

    if poorly_calibrated:
        base_pipe = make_full_pipeline(numeric_cols, binary_cols, categorical_cols, candidates[best_name])
        calibrated_model = CalibratedClassifierCV(base_pipe, method="sigmoid", cv=5)
        calibrated_model.fit(X_train, y_train)
        y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
        _, _, brier_after = check_calibration(y_test, y_prob_calibrated)

        if brier_after < brier_before:
            best_pipe = calibrated_model
            y_prob_final = y_prob_calibrated
            calibration_applied = True

    score = np.round(y_prob_final * 100).astype(int)
    hot_threshold, precision_at_hot, recall_at_hot, target_met = find_threshold(y_test, score)
    warm_threshold = max(hot_threshold - 20, 0)

    joblib.dump(best_pipe, MODEL_DIR / "pipeline.joblib")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": best_name,
        "test_roc_auc": test_roc_auc,
        "hot_threshold": hot_threshold,
        "warm_threshold": warm_threshold,
        "precision_at_hot_threshold": precision_at_hot,
        "recall_at_hot_threshold": recall_at_hot,
        "precision_target_met": target_met,
        "calibration_applied": calibration_applied,
        "calibration_check": {
            "max_reliability_gap_before": max_gap,
            "brier_score_before": brier_before,
            "brier_score_after": brier_after,
        },
        "random_state": RANDOM_STATE,
        "feature_columns": feature_columns,
        "candidate_model_comparison": cv_results,
    }

    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(MODEL_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)

    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Calibration applied: {calibration_applied} (max reliability gap: {max_gap:.4f})")
    print(f"Hot threshold: {hot_threshold} (precision {precision_at_hot:.3f}, recall {recall_at_hot:.3f})")
    print(f"Precision target (>=80%) met: {target_met}")
    print(f"Warm threshold: {warm_threshold}")
    print("Saved model/pipeline.joblib, model/metadata.json, model/feature_importance.json")


if __name__ == "__main__":
    main()
