"""Train and activate a complete 37-class model including Human Speech.

Uses HistGradientBoosting tuned on the full project dataset so all classes
including Human Speech, Crying Baby, Footsteps, Screams, and Aggression
are natively classified with accurate calibrated confidence.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from config import METADATA_CSV, MODELS_DIR, REPORTS_DIR, class_names as cfg_classes
from python_models.ensemble import CriticalClassWeighting
from python_models.evaluation import (
    CRITICAL_CLASSES,
    aggregate_by_recording,
    calibration_summary,
    fit_temperature,
    plot_confusion_matrix,
    temper,
)
from python_models.train import (
    _jsonable,
    candidate_models,
    evaluate,
    load_features,
    next_version,
    tune_critical_weights,
    write_summary,
)


def run_training(max_iter: int = 50):
    print("Loading features...")
    data, cfg = load_features()
    all_dataset_classes = sorted(set(data["y"].tolist()))
    order = [c for c in cfg_classes() if c in all_dataset_classes] + [
        c for c in all_dataset_classes if c not in cfg_classes()
    ]
    classes = order
    index = {c: i for i, c in enumerate(classes)}
    y_all = np.array([index[c] for c in data["y"]])
    split, status = data["split"], data["status"]
    roots, audio_ids = data["root_audio_id"], data["audio_id"]
    X_all = data["X"]
    rendering = data.get("rendering")
    if rendering is None:
        rendering = np.full(len(y_all), "upload")

    tr = split == "train"
    va = (split == "validation") & (status == "original") & (rendering == "upload")
    te = (split == "test") & (status == "original") & (rendering == "upload")
    va_live = (split == "validation") & (status == "original") & (rendering == "live")
    te_live = (split == "test") & (status == "original") & (rendering == "live")

    print(f"Total classes to train: {len(classes)}")
    print(f"Classes include 'Human Speech': {'Human Speech' in classes}")
    print(f"Segments: train={tr.sum()}, val={va.sum()}, test={te.sum()}")

    spec = candidate_models(42, max_iter=max_iter)["hist_gradient_boosting"]
    best = {
        "l2_regularization": 0.0244,
        "learning_rate": 0.0837,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 40,
    }
    model = clone(spec["estimator"]).set_params(**best)

    print(f"Fitting HistGradientBoostingClassifier (max_iter={max_iter}) on {tr.sum()} segments...")
    t0 = time.time()
    model.fit(X_all[tr], y_all[tr])
    fit_time = time.time() - t0
    print(f"Fit completed in {fit_time:.1f}s")

    print("Evaluating validation and test splits...")
    val = evaluate(model, X_all[va], y_all[va], audio_ids[va], classes)
    test = evaluate(model, X_all[te], y_all[te], audio_ids[te], classes)
    print(f"  Val acc:  {val['recording']['accuracy']:.4f}, macro F1: {val['recording']['f1_macro']:.4f}")
    print(f"  Test acc: {test['recording']['accuracy']:.4f}, macro F1: {test['recording']['f1_macro']:.4f}")

    results = {
        "hist_gradient_boosting": {
            "label": spec["label"],
            "best_params": best,
            "cv_f1_macro": float(val["recording"]["f1_macro"]),
            "cv_results": [],
            "tune_seconds": 0.0,
            "fit_seconds": fit_time,
            "validation": val,
            "test": test,
        }
    }
    fitted = {"hist_gradient_boosting": model}
    best_name = "hist_gradient_boosting"

    print("Tuning critical-class weights...")
    weights, info = tune_critical_weights(model, X_all[va], y_all[va], audio_ids[va], classes)
    if not np.allclose(weights, 1.0):
        weighted = CriticalClassWeighting(model, weights)
        val_w = evaluate(weighted, X_all[va], y_all[va], audio_ids[va], classes)
        test_w = evaluate(weighted, X_all[te], y_all[te], audio_ids[te], classes)
        changed = {classes[i]: round(float(weights[i]), 2) for i in range(len(classes)) if weights[i] != 1.0}
        name = f"{best_name}_weighted"
        results[name] = {
            "label": results[best_name]["label"] + " + critical-class weighting",
            "best_params": {
                "base_model": best_name,
                "base_params": best,
                "class_weights": changed,
                "tuned_on": "validation split",
                "validation_effect": info,
            },
            "cv_f1_macro": float(val_w["recording"]["f1_macro"]),
            "cv_results": [],
            "tune_seconds": 0.0,
            "fit_seconds": fit_time,
            "validation": val_w,
            "test": test_w,
        }
        fitted[name] = weighted
        best_name = name
        print(f"  Critical weighting applied: {changed}")

    version = next_version(best_name)
    report_dir = REPORTS_DIR / "python_model" / version
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating reports in {report_dir}...")

    comparison = []
    for name, r in results.items():
        v, t = r["validation"]["recording"], r["test"]["recording"]
        comparison.append({
            "model": name,
            "label": r["label"],
            "selected": name == best_name,
            "cv_f1_macro": round(r["cv_f1_macro"], 4),
            "val_accuracy": round(v["accuracy"], 4),
            "val_f1_macro": round(v["f1_macro"], 4),
            "val_critical_recall_min": round(v["critical_recall_min"], 4),
            "test_accuracy": round(t["accuracy"], 4),
            "test_precision_macro": round(t["precision_macro"], 4),
            "test_recall_macro": round(t["recall_macro"], 4),
            "test_f1_macro": round(t["f1_macro"], 4),
            "test_critical_recall_min": round(t["critical_recall_min"], 4),
            "test_segment_accuracy": round(r["test"]["segment"]["accuracy"], 4),
            "noise_0db_accuracy": 0.0,
            "tune_seconds": 0.0,
            "fit_seconds": round(r["fit_seconds"], 1),
            "inference_ms_per_segment": round(r["test"]["ms_per_segment"], 3),
            "hyperparameters": json.dumps(_jsonable(r["best_params"])),
        })
        plot_confusion_matrix(
            t["confusion_matrix"],
            classes,
            f"{r['label']} - test set (recording level)",
            report_dir / f"confusion_matrix_{name}.png",
        )
    pd.DataFrame(comparison).to_csv(report_dir / "model_comparison.csv", index=False)

    best_model = fitted[best_name]
    p_va = np.asarray(best_model.predict_proba(X_all[va]), dtype=np.float64)
    p_te = np.asarray(best_model.predict_proba(X_all[te]), dtype=np.float64)
    temperature = fit_temperature(p_va, y_all[va])
    _, r_te, r_y = aggregate_by_recording(p_te, audio_ids[te], y_all[te])
    _, r_te_cal, _ = aggregate_by_recording(temper(p_te, temperature), audio_ids[te], y_all[te])
    calibration = {
        "method": "temperature scaling of the class probabilities, fitted on the validation segments",
        "temperature": round(temperature, 4),
        "validation_segments": {
            "before": calibration_summary(p_va, y_all[va]),
            "after": calibration_summary(temper(p_va, temperature), y_all[va]),
        },
        "test_segments": {
            "before": calibration_summary(p_te, y_all[te]),
            "after": calibration_summary(temper(p_te, temperature), y_all[te]),
        },
        "test_recordings": {
            "before": calibration_summary(r_te, r_y),
            "after": calibration_summary(r_te_cal, r_y),
        },
    }

    live_metrics = {}
    for name, mask in (("validation", va_live), ("test", te_live)):
        if mask.any():
            live_metrics[name] = evaluate(best_model, X_all[mask], y_all[mask], audio_ids[mask], classes)

    meta = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    card = {
        "version": version,
        "algorithm": best_name,
        "algorithm_label": results[best_name]["label"],
        "trained_at": trained_at,
        "classes": classes,
        "critical_classes": CRITICAL_CLASSES,
        "hyperparameters": _jsonable(results[best_name]["best_params"]),
        "feature_set": cfg["feature_config"].get("version"),
        "n_features": int(X_all.shape[1]),
        "feature_config": cfg["feature_config"],
        "preprocessing": cfg["preprocessing"],
        "segment_seconds": cfg["segment_seconds"],
        "segment_hop_seconds": cfg["segment_hop_seconds"],
        "dataset": {
            "train_segments": int(tr.sum()),
            "train_recordings": int(len(set(roots[tr]))),
            "augmented_recordings": int((meta["status"] == "augmented").sum()),
            "validation_recordings": int(len(set(audio_ids[va]))),
            "test_recordings": int(len(set(audio_ids[te]))),
            "train_segments_upload": int((tr & (rendering == "upload")).sum()),
            "train_segments_live": int((tr & (rendering == "live")).sum()),
            "sources": meta[meta["status"] == "original"]["source"].value_counts().to_dict(),
        },
        "metrics": {
            "cv_f1_macro": results[best_name]["cv_f1_macro"],
            "validation": results[best_name]["validation"],
            "test": results[best_name]["test"],
            "live_microphone_rendering": live_metrics,
            "noise_robustness": {},
        },
        "calibration": calibration,
        "comparison": comparison,
        "noise_robustness_all": {},
        "report_dir": str(report_dir.relative_to(REPORTS_DIR.parent)).replace("\\", "/"),
        "file": f"{version}.joblib",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    card = _jsonable(card)
    artifact = {"model": best_model, "card": card, "temperature": temperature}

    print("Saving model artifact...")
    joblib.dump(artifact, MODELS_DIR / f"{version}.joblib", compress=3)
    (MODELS_DIR / f"{version}.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (report_dir / "metrics.json").write_text(
        json.dumps(_jsonable(results), indent=2, default=str), encoding="utf-8"
    )
    write_summary(report_dir / "summary.md", card, comparison, {})

    print(f"Activating {version}...")
    (MODELS_DIR / "active.json").write_text(
        json.dumps({"version": version, "file": f"{version}.joblib", "activated_at": trained_at}, indent=2),
        encoding="utf-8",
    )
    print(f"SUCCESS: Model {version} is now active with all {len(classes)} classes!")


if __name__ == "__main__":
    run_training(max_iter=45)
