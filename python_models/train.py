"""Train, tune, compare and save the Python sound classifier.

    python -m python_models.train                # full run (about 20-40 minutes on 4 cores)
    python -m python_models.train --quick        # small search for development
    python -m python_models.train --models svm random_forest

Pipeline
1. Load ``data/features/features.npz`` (one row per segment; every segment
   carries its recording's Audio ID, class and split).
2. Training data = train split (originals + augmented copies). Validation and
   test data = original recordings of those splits only; they are never used
   for fitting.
3. Hyper-parameter search per model: ``RandomizedSearchCV`` with
   ``GroupKFold`` (groups = root Audio ID, so segments / augmentations of one
   recording never sit on both sides of a fold), scoring = macro F1.
4. The tuned models are compared on the validation split at recording level
   (mean of segment probabilities). The best macro F1 wins; critical-class
   recall breaks ties.
5. All models are then evaluated on the unseen test split (accuracy,
   precision, recall, F1, macro F1, confusion matrix, class-wise results,
   critical-event recall) and on noisy copies of the test recordings
   (noise-robustness at 20 / 10 / 5 / 0 dB SNR).
6. The selected model is saved to ``python_models/saved/<version>.joblib``
   with a model card, and ``active.json`` points the web app to it.
   Reports go to ``reports/python_model/<version>/``.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import DATASET_DIR, FEATURES_DIR, METADATA_CSV, MODELS_DIR, REPORTS_DIR
from python_models.ensemble import CriticalClassWeighting, SoftVotingEnsemble
from python_models.evaluation import (
    CRITICAL_CLASSES,
    aggregate_by_recording,
    calibration_summary,
    classification_metrics,
    fit_temperature,
    plot_confusion_matrix,
    temper,
)

try:  # XGBoost is optional
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


# --------------------------------------------------------------------------- #
# Candidate models and search spaces
# --------------------------------------------------------------------------- #
def _calibrated_svm(est, seed: int):
    """Final SVM with probability outputs (Platt scaling via CalibratedClassifierCV)."""
    svc = est.named_steps["clf"]
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", CalibratedClassifierCV(SVC(kernel="rbf", C=svc.C, gamma=svc.gamma, class_weight="balanced",
                                           random_state=seed), method="sigmoid", cv=3, ensemble=False)),
    ])


def candidate_models(seed: int, max_iter: int | None = None) -> dict:
    """The models the run compares.

    ``max_iter`` caps the boosting rounds. Cost grows with the number of classes -
    a gradient-boosting round fits one tree per class - so a 37-class dataset needs
    a smaller budget than the default to train in a sensible time.
    """
    boost_iter = max_iter or 400
    models = {
        "svm": {
            "label": "Support Vector Machine (RBF kernel)",
            "estimator": Pipeline([("scale", StandardScaler()),
                                   ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed))]),
            "params": {"clf__C": loguniform(0.5, 100), "clf__gamma": loguniform(1e-4, 3e-2)},
            "search_jobs": -1,
            "finalize": lambda est: _calibrated_svm(est, seed),
        },
        "random_forest": {
            "label": "Random Forest",
            "estimator": RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
            "params": {
                "n_estimators": randint(200, 500),
                "max_depth": [None, 16, 24, 32],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.2, 0.35],
            },
            "search_jobs": 1,
        },
        "hist_gradient_boosting": {
            "label": "Histogram Gradient Boosting",
            "estimator": HistGradientBoostingClassifier(max_iter=boost_iter, early_stopping=True,
                                                        validation_fraction=0.1, random_state=seed),
            "params": {
                "learning_rate": loguniform(0.03, 0.3),
                "max_leaf_nodes": [15, 31, 63],
                "l2_regularization": loguniform(1e-4, 1.0),
                "min_samples_leaf": [10, 20, 40],
            },
            "search_jobs": 1,
        },
        "mlp": {
            "label": "Multi-layer Perceptron (neural network)",
            "estimator": Pipeline([("scale", StandardScaler()),
                                   ("clf", MLPClassifier(early_stopping=True, max_iter=300, random_state=seed))]),
            "params": {
                "clf__hidden_layer_sizes": [(256,), (256, 128), (512, 256)],
                "clf__alpha": loguniform(1e-5, 1e-2),
                "clf__learning_rate_init": loguniform(3e-4, 3e-3),
            },
            "search_jobs": -1,
        },
    }
    if XGBClassifier is not None:
        models["xgboost"] = {
            "label": "XGBoost (gradient-boosted trees)",
            "estimator": XGBClassifier(tree_method="hist", eval_metric="mlogloss", n_jobs=-1, random_state=seed),
            "params": {
                "n_estimators": randint(200, 500),
                "max_depth": [4, 6, 8],
                "learning_rate": loguniform(0.03, 0.3),
                "subsample": uniform(0.7, 0.3),
                "colsample_bytree": uniform(0.5, 0.4),
                "min_child_weight": [1, 3, 5],
            },
            "search_jobs": 1,
        }
    return models


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_features():
    data = np.load(FEATURES_DIR / "features.npz", allow_pickle=True)
    cfg = json.loads((FEATURES_DIR / "feature_config.json").read_text())
    return {k: data[k] for k in data.files}, cfg


def tuning_subset(roots: np.ndarray, y: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Recording-level stratified subsample (keeps whole recordings together)."""
    if len(y) <= max_samples:
        return np.arange(len(y))
    per_class = max(1, max_samples // len(np.unique(y)))
    keep = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rs = rng.permutation(np.unique(roots[idx]))
        count = 0
        for r in rs:
            sel = idx[roots[idx] == r]
            keep[sel] = True
            count += len(sel)
            if count >= per_class:
                break
    return np.flatnonzero(keep)


def evaluate(model, X, y, groups, classes) -> dict:
    t0 = time.perf_counter()
    proba = model.predict_proba(X)
    ms_per_segment = (time.perf_counter() - t0) / max(1, len(X)) * 1000
    seg = classification_metrics(y, proba.argmax(axis=1), classes)
    _, agg, truth = aggregate_by_recording(proba, groups, y)
    rec = classification_metrics(truth, agg.argmax(axis=1), classes)
    return {"segment": seg, "recording": rec, "ms_per_segment": ms_per_segment}


def tune_critical_weights(model, X, y, groups, classes, max_f1_drop: float = 0.01):
    """Coordinate search for per-class probability weights on the validation split.

    Objective (compared as a tuple): (minimum critical-class recall, macro F1),
    subject to macro F1 >= unweighted macro F1 - ``max_f1_drop``.
    """
    proba = np.asarray(model.predict_proba(X), dtype=float)
    _, agg, truth = aggregate_by_recording(proba, groups, y)
    critical = [i for i, c in enumerate(classes) if c in CRITICAL_CLASSES]

    def score(w):
        p = agg * w
        m = classification_metrics(truth, p.argmax(axis=1), classes)
        return m["critical_recall_min"], m["f1_macro"]

    weights = np.ones(len(classes))
    base_recall, base_f1 = score(weights)
    best = (base_recall, base_f1)
    grid = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    for _ in range(3):
        improved = False
        for c in critical:
            for g in grid:
                trial = weights.copy()
                trial[c] = g
                r, f = score(trial)
                if f < base_f1 - max_f1_drop:
                    continue
                if (round(r, 4), round(f, 4)) > (round(best[0], 4), round(best[1], 4)):
                    best, weights, improved = (r, f), trial, True
        if not improved:
            break
    return weights, {"before": {"critical_recall_min": base_recall, "f1_macro": base_f1},
                     "after": {"critical_recall_min": best[0], "f1_macro": best[1]}}


# --------------------------------------------------------------------------- #
# Noise robustness
# --------------------------------------------------------------------------- #
def _noisy_features(row: dict, noise: np.ndarray, snr_db: float, pre_cfg, seg_s, hop_s, feat_cfg):
    from audio_preprocessing import preprocess_audio, segment_audio, select_event_segments
    from audio_preprocessing.loader import load_mono
    from feature_extraction.features import extract_features

    y, sr = load_mono(DATASET_DIR / row["relative_path"], pre_cfg.target_sr)
    if len(noise) < len(y):
        noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))
    n = noise[: len(y)]
    frame = 1024
    k = len(y) // frame
    e = np.mean(y[: k * frame].reshape(k, frame) ** 2, axis=1) if k >= 2 else np.array([np.mean(y ** 2)])
    ps = float(np.mean(e[e >= 0.01 * e.max()])) + 1e-12
    y = y + np.sqrt(ps / (np.mean(n ** 2) + 1e-12) / 10 ** (snr_db / 10)) * n
    y = y / max(1.0, float(np.max(np.abs(y))))
    pre = preprocess_audio(y, sr, pre_cfg)
    segs = segment_audio(pre.y, pre.sr, seg_s, hop_s)
    if row["class_label"] != "Background Noise":
        segs = select_event_segments(segs)
    return [extract_features(s.audio, feat_cfg) for s in segs[:8]], row["audio_id"], row["class_label"]


def noise_robustness(models: dict, classes: list[str], cfg: dict, levels=(20, 10, 5, 0), jobs: int = -1) -> dict:
    from joblib import Parallel, delayed

    from audio_preprocessing import PreprocessConfig
    from audio_preprocessing.loader import load_mono
    from feature_extraction.features import FeatureConfig

    meta = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    test = meta[(meta["split"] == "test") & (meta["status"] == "original")].to_dict("records")
    bank_rows = meta[(meta["split"] == "train") & (meta["status"] == "original") & (meta["class_label"] == "Background Noise")].head(40)
    pre_cfg = PreprocessConfig(**cfg["preprocessing"])
    feat_cfg = FeatureConfig.from_dict(cfg["feature_config"])
    bank = [load_mono(DATASET_DIR / p, pre_cfg.target_sr)[0] for p in bank_rows["relative_path"]]
    if not bank:
        return {}
    index = {c: i for i, c in enumerate(classes)}
    results: dict = {name: {} for name in models}
    rng = np.random.default_rng(0)
    for snr in levels:
        picks = [bank[int(rng.integers(0, len(bank)))] for _ in test]
        out = Parallel(n_jobs=jobs)(
            delayed(_noisy_features)(r, nz, snr, pre_cfg, cfg["segment_seconds"], cfg["segment_hop_seconds"], feat_cfg)
            for r, nz in zip(test, picks)
        )
        X, groups, y = [], [], []
        for feats, aid, label in out:
            for f in feats:
                X.append(f)
                groups.append(aid)
                y.append(index[label])
        X, groups, y = np.vstack(X), np.array(groups), np.array(y)
        for name, model in models.items():
            proba = model.predict_proba(X)
            _, agg, truth = aggregate_by_recording(proba, groups, y)
            m = classification_metrics(truth, agg.argmax(axis=1), classes)
            results[name][f"{snr} dB"] = {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                                          "critical_recall_min": m["critical_recall_min"]}
        print(f"  noise robustness @ {snr:>2} dB SNR: " + ", ".join(
            f"{n}={results[n][f'{snr} dB']['accuracy']:.3f}" for n in models))
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def next_version(algorithm: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in MODELS_DIR.glob("py-v*.json"):
        try:
            nums.append(int(p.stem.split("-")[1][1:]))
        except (IndexError, ValueError):
            pass
    name = algorithm.replace("_", "-").replace("hist-gradient-boosting", "hgb")
    return f"py-v{(max(nums) + 1) if nums else 1}-{name}"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", help="subset of: svm random_forest hist_gradient_boosting xgboost mlp")
    parser.add_argument("--n-iter", type=int, default=6, help="random-search iterations per model")
    parser.add_argument("--tune-samples", type=int, default=5000, help="max segments used for the search")
    parser.add_argument("--quick", action="store_true", help="3 iterations, 2500 samples, no noise test")
    parser.add_argument("--no-noise", action="store_true", help="skip the noise-robustness evaluation")
    parser.add_argument("--max-iter", type=int,
                        help="cap the boosting rounds (default 400). A gradient-boosting round fits one "
                             "tree per class, so a run with many classes trains far faster with a lower cap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-activate", action="store_true", help="save but do not make the model active")
    parser.add_argument("--params-from", help="reuse the tuned hyper-parameters in <report dir>/metrics.json "
                                              "instead of searching again (e.g. after adding data)")
    parser.add_argument("--no-ensemble", action="store_true", help="do not build the soft-voting ensemble")
    parser.add_argument("--no-critical-weighting", action="store_true",
                        help="do not tune decision weights for critical classes")
    args = parser.parse_args(argv)
    if args.quick:
        args.n_iter, args.tune_samples, args.no_noise = 3, 2500, True
    previous: dict = {}
    if args.params_from:
        from pathlib import Path

        prev_path = Path(args.params_from)
        prev_path = prev_path / "metrics.json" if prev_path.is_dir() else prev_path
        previous = json.loads(prev_path.read_text(encoding="utf-8"))
        print(f"Reusing tuned hyper-parameters from {prev_path}")

    data, cfg = load_features()
    classes = sorted(set(data["y"].tolist()))
    from config import class_names as cfg_classes

    order = [c for c in cfg_classes() if c in classes] + [c for c in classes if c not in cfg_classes()]
    classes = order
    index = {c: i for i, c in enumerate(classes)}
    y_all = np.array([index[c] for c in data["y"]])
    split, status = data["split"], data["status"]
    roots, audio_ids = data["root_audio_id"], data["audio_id"]
    X_all = data["X"]

    # Training uses both renderings (upload and live microphone); selection and the headline numbers use the
    # upload rendering, and the live rendering of the same recordings is reported separately.
    rendering = data.get("rendering")
    if rendering is None:
        rendering = np.full(len(y_all), "upload")
    tr = split == "train"
    va = (split == "validation") & (status == "original") & (rendering == "upload")
    te = (split == "test") & (status == "original") & (rendering == "upload")
    va_live = (split == "validation") & (status == "original") & (rendering == "live")
    te_live = (split == "test") & (status == "original") & (rendering == "live")
    # Leakage guard: a root recording may only appear in one split.
    assert not (set(roots[tr]) & set(roots[va])), "train/validation leakage"
    assert not (set(roots[tr]) & set(roots[te])), "train/test leakage"
    assert not (set(roots[va]) & set(roots[te])), "validation/test leakage"

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Segments: train={tr.sum()} (recordings {len(set(roots[tr]))}), "
          f"validation={va.sum()} ({len(set(audio_ids[va]))}), test={te.sum()} ({len(set(audio_ids[te]))})")
    print(f"  renderings: train upload={int((tr & (rendering == 'upload')).sum())} "
          f"live={int((tr & (rendering == 'live')).sum())}; "
          f"live evaluation windows: validation={int(va_live.sum())}, test={int(te_live.sum())}")

    rng = np.random.default_rng(args.seed)
    tune_idx = np.flatnonzero(tr)[tuning_subset(roots[tr], y_all[tr], args.tune_samples, rng)]
    specs = candidate_models(args.seed, args.max_iter)
    selected = args.models or list(specs)
    results: dict = {}
    fitted: dict = {}

    for name in selected:
        if name not in specs:
            print(f"Skipping unknown model '{name}'")
            continue
        spec = specs[name]
        print(f"\n=== {spec['label']} ===")
        t0 = time.time()
        if name in previous:
            best = {k: (tuple(v) if isinstance(v, list) else v) for k, v in previous[name]["best_params"].items()}
            cv_score = float(previous[name]["cv_f1_macro"])
            cv_records = previous[name].get("cv_results", [])
            print(f"  reused parameters (CV macro F1 {cv_score:.4f}): {best}")
        else:
            search = RandomizedSearchCV(
                spec["estimator"], spec["params"], n_iter=args.n_iter, scoring="f1_macro",
                cv=GroupKFold(n_splits=3), n_jobs=spec["search_jobs"], random_state=args.seed, refit=False,
            )
            search.fit(X_all[tune_idx], y_all[tune_idx], groups=roots[tune_idx])
            best = search.best_params_
            cv_score = float(search.best_score_)
            cv_table = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "mean_fit_time"]]
            cv_records = cv_table.sort_values("mean_test_score", ascending=False).to_dict("records")
            print(f"  best CV macro F1 = {cv_score:.4f} with {best}  ({time.time() - t0:.0f} s)")
        tune_time = time.time() - t0
        from sklearn.base import clone

        model = clone(spec["estimator"]).set_params(**best)
        if "finalize" in spec:
            model = spec["finalize"](model)
        t1 = time.time()
        model.fit(X_all[tr], y_all[tr])
        fit_time = time.time() - t1
        val = evaluate(model, X_all[va], y_all[va], audio_ids[va], classes)
        test = evaluate(model, X_all[te], y_all[te], audio_ids[te], classes)
        print(f"  validation: acc={val['recording']['accuracy']:.4f} macroF1={val['recording']['f1_macro']:.4f} "
              f"critical-recall(min)={val['recording']['critical_recall_min']:.3f}")
        print(f"  test:       acc={test['recording']['accuracy']:.4f} macroF1={test['recording']['f1_macro']:.4f} "
              f"critical-recall(min)={test['recording']['critical_recall_min']:.3f}")
        results[name] = {
            "label": spec["label"],
            "best_params": best,
            "cv_f1_macro": cv_score,
            "cv_results": cv_records,
            "tune_seconds": tune_time,
            "fit_seconds": fit_time,
            "validation": val,
            "test": test,
        }
        fitted[name] = model

    if not fitted:
        print("No models trained.")
        return 1

    # Soft-voting ensemble of the three best models on the validation split.
    if not args.no_ensemble and len(fitted) >= 3:
        top = sorted(fitted, key=lambda n: results[n]["validation"]["recording"]["f1_macro"], reverse=True)[:3]
        ens = SoftVotingEnsemble([(n, fitted[n]) for n in top])
        val = evaluate(ens, X_all[va], y_all[va], audio_ids[va], classes)
        test = evaluate(ens, X_all[te], y_all[te], audio_ids[te], classes)
        label = "Soft-voting ensemble (" + " + ".join(results[n]["label"].split(" (")[0] for n in top) + ")"
        print(f"\n=== {label} ===")
        print(f"  validation: acc={val['recording']['accuracy']:.4f} macroF1={val['recording']['f1_macro']:.4f} "
              f"critical-recall(min)={val['recording']['critical_recall_min']:.3f}")
        print(f"  test:       acc={test['recording']['accuracy']:.4f} macroF1={test['recording']['f1_macro']:.4f} "
              f"critical-recall(min)={test['recording']['critical_recall_min']:.3f}")
        results["ensemble"] = {
            "label": label,
            "best_params": {"members": top, "weights": "equal", "combination": "mean of class probabilities"},
            "cv_f1_macro": float(np.mean([results[n]["cv_f1_macro"] for n in top])),
            "cv_results": [],
            "tune_seconds": 0.0,
            "fit_seconds": float(sum(results[n]["fit_seconds"] for n in top)),
            "validation": val,
            "test": test,
        }
        fitted["ensemble"] = ens

    def rank_key(n):
        v = results[n]["validation"]["recording"]
        return (round(v["f1_macro"], 3), v["critical_recall_min"])

    best_name = max(fitted, key=rank_key)
    print(f"\nSelected model: {results[best_name]['label']} (validation macro F1 "
          f"{results[best_name]['validation']['recording']['f1_macro']:.4f})")

    # Critical-class decision weighting, tuned on the validation split only.
    if not args.no_critical_weighting:
        weights, info = tune_critical_weights(fitted[best_name], X_all[va], y_all[va], audio_ids[va], classes)
        if not np.allclose(weights, 1.0):
            weighted = CriticalClassWeighting(fitted[best_name], weights)
            val_w = evaluate(weighted, X_all[va], y_all[va], audio_ids[va], classes)
            test_w = evaluate(weighted, X_all[te], y_all[te], audio_ids[te], classes)
            changed = {classes[i]: round(float(weights[i]), 2) for i in range(len(classes)) if weights[i] != 1.0}
            name = f"{best_name}_weighted"
            results[name] = {
                "label": results[best_name]["label"] + " + critical-class weighting",
                "best_params": {"base_model": best_name, "base_params": results[best_name]["best_params"],
                                "class_weights": changed, "tuned_on": "validation split",
                                "validation_effect": info},
                "cv_f1_macro": results[best_name]["cv_f1_macro"],
                "cv_results": [],
                "tune_seconds": 0.0,
                "fit_seconds": results[best_name]["fit_seconds"],
                "validation": val_w,
                "test": test_w,
            }
            fitted[name] = weighted
            print(f"Critical-class weighting {changed}: validation min critical recall "
                  f"{info['before']['critical_recall_min']:.3f} -> {info['after']['critical_recall_min']:.3f}, "
                  f"macro F1 {info['before']['f1_macro']:.3f} -> {info['after']['f1_macro']:.3f}")
            print(f"  test: acc={test_w['recording']['accuracy']:.4f} macroF1={test_w['recording']['f1_macro']:.4f} "
                  f"critical-recall(min)={test_w['recording']['critical_recall_min']:.3f}")
            best_name = name
        else:
            print("Critical-class weighting: no weights improved validation critical recall; kept the model as is.")

    robustness = {}
    if not args.no_noise:
        print("\nNoise-robustness evaluation on the test split ...")
        robustness = noise_robustness(fitted, classes, cfg)

    version = next_version(best_name)
    report_dir = REPORTS_DIR / "python_model" / version
    report_dir.mkdir(parents=True, exist_ok=True)

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
            "noise_0db_accuracy": round(robustness.get(name, {}).get("0 dB", {}).get("accuracy", float("nan")), 4),
            "tune_seconds": round(r["tune_seconds"], 1),
            "fit_seconds": round(r["fit_seconds"], 1),
            "inference_ms_per_segment": round(r["test"]["ms_per_segment"], 3),
            "hyperparameters": json.dumps(_jsonable(r["best_params"])),
        })
        plot_confusion_matrix(t["confusion_matrix"], classes, f"{r['label']} - test set (recording level)",
                              report_dir / f"confusion_matrix_{name}.png")
    pd.DataFrame(comparison).to_csv(report_dir / "model_comparison.csv", index=False)

    best_model = fitted[best_name]
    # Confidence calibration: one temperature fitted on the validation segments (log loss). It changes the
    # confidence values shown in the app, never the predicted class.
    p_va = np.asarray(best_model.predict_proba(X_all[va]), dtype=np.float64)
    p_te = np.asarray(best_model.predict_proba(X_all[te]), dtype=np.float64)
    temperature = fit_temperature(p_va, y_all[va])
    _, r_te, r_y = aggregate_by_recording(p_te, audio_ids[te], y_all[te])
    _, r_te_cal, _ = aggregate_by_recording(temper(p_te, temperature), audio_ids[te], y_all[te])
    calibration = {
        "method": "temperature scaling of the class probabilities, fitted on the validation segments",
        "temperature": round(temperature, 4),
        "validation_segments": {"before": calibration_summary(p_va, y_all[va]),
                                "after": calibration_summary(temper(p_va, temperature), y_all[va])},
        "test_segments": {"before": calibration_summary(p_te, y_all[te]),
                          "after": calibration_summary(temper(p_te, temperature), y_all[te])},
        "test_recordings": {"before": calibration_summary(r_te, r_y), "after": calibration_summary(r_te_cal, r_y)},
    }
    print(f"Calibration: temperature {temperature:.3f}; test segment ECE "
          f"{calibration['test_segments']['before']['ece']:.3f} -> {calibration['test_segments']['after']['ece']:.3f}")
    # The same model on the live-microphone rendering of the validation and test recordings.
    live_metrics = {}
    for name, mask in (("validation", va_live), ("test", te_live)):
        if mask.any():
            live_metrics[name] = evaluate(best_model, X_all[mask], y_all[mask], audio_ids[mask], classes)
    if live_metrics:
        lt = live_metrics.get("test", {}).get("recording", {})
        print(f"Live-microphone rendering (test recordings): acc={lt.get('accuracy', 0):.4f} "
              f"macroF1={lt.get('f1_macro', 0):.4f} critical-recall(min)={lt.get('critical_recall_min', 0):.3f}")
    # Sample predictions (first 3 test recordings of every class), with the calibrated confidences.
    proba = temper(p_te, temperature)
    ids, agg, truth = aggregate_by_recording(proba, audio_ids[te], y_all[te])
    samples = []
    for c in range(len(classes)):
        for i in np.flatnonzero(truth == c)[:3]:
            top = np.argsort(agg[i])[::-1][:3]
            samples.append({
                "audio_id": ids[i],
                "actual": classes[truth[i]],
                "predicted": classes[top[0]],
                "confidence": round(float(agg[i][top[0]]), 4),
                "top3": "; ".join(f"{classes[j]} {agg[i][j]:.2f}" for j in top),
                "correct": bool(top[0] == truth[i]),
            })
    pd.DataFrame(samples).to_csv(report_dir / "sample_predictions.csv", index=False)

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
            "noise_robustness": robustness.get(best_name, {}),
        },
        "calibration": calibration,
        "comparison": comparison,
        "noise_robustness_all": robustness,
        "report_dir": str(report_dir.relative_to(REPORTS_DIR.parent)).replace("\\", "/"),
        "file": f"{version}.joblib",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    card = _jsonable(card)
    artifact = {"model": best_model, "card": card, "temperature": temperature}
    joblib.dump(artifact, MODELS_DIR / f"{version}.joblib", compress=3)
    (MODELS_DIR / f"{version}.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(_jsonable(results), indent=2, default=str), encoding="utf-8")
    (report_dir / "noise_robustness.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")
    write_summary(report_dir / "summary.md", card, comparison, robustness)
    if not args.no_activate:
        (MODELS_DIR / "active.json").write_text(json.dumps({"version": version, "file": f"{version}.joblib",
                                                             "activated_at": trained_at}, indent=2), encoding="utf-8")
    print(f"\nSaved {MODELS_DIR / (version + '.joblib')}\nReports in {report_dir}")
    return 0


def write_summary(path, card: dict, comparison: list[dict], robustness: dict) -> None:
    t = card["metrics"]["test"]["recording"]
    lines = [
        f"# Python model {card['version']}",
        "",
        f"* Algorithm: **{card['algorithm_label']}**",
        f"* Trained: {card['trained_at']}",
        f"* Features: {card['n_features']} ({card['feature_set']}), segments of {card['segment_seconds']} s",
        f"* Hyper-parameters: `{json.dumps(card['hyperparameters'])}`",
        "",
        "## Test set (recording level)",
        "",
        f"| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |",
        "|---|---|---|---|---|",
        f"| {t['accuracy']:.3f} | {t['precision_macro']:.3f} | {t['recall_macro']:.3f} | {t['f1_macro']:.3f} | {t['critical_recall_min']:.3f} |",
        "",
        "## Class-wise performance (test)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for cls, m in t["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    lines += ["", "## Model comparison", "",
              "| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |",
              "|---|---|---|---|---|---|---|"]
    for c in comparison:
        lines.append(f"| {c['label']} | {c['cv_f1_macro']:.3f} | {c['val_f1_macro']:.3f} | {c['test_accuracy']:.3f} | "
                     f"{c['test_f1_macro']:.3f} | {c['test_critical_recall_min']:.3f} | {'yes' if c['selected'] else ''} |")
    live = card["metrics"].get("live_microphone_rendering") or {}
    if live:
        lines += ["", "## Live-microphone rendering (same recordings, as the live monitor delivers them)", "",
                  "| Split | Accuracy | Macro F1 | Min critical recall |", "|---|---|---|---|"]
        for split, m in live.items():
            r = m["recording"]
            lines.append(f"| {split} | {r['accuracy']:.3f} | {r['f1_macro']:.3f} | {r['critical_recall_min']:.3f} |")
    cal = card.get("calibration")
    if cal:
        b, a = cal["test_segments"]["before"], cal["test_segments"]["after"]
        lines += ["", "## Confidence calibration", "",
                  f"Temperature {cal['temperature']} fitted on the validation segments. Expected calibration error "
                  f"on the test segments {b['ece']} -> {a['ece']}, mean confidence {b['mean_confidence']} -> "
                  f"{a['mean_confidence']} against accuracy {a['accuracy']}. The predicted class does not change."]
    if robustness:
        levels = list(next(iter(robustness.values())).keys())
        lines += ["", "## Noise robustness (test accuracy with added background noise)", "",
                  "| Model | " + " | ".join(levels) + " |", "|---" * (len(levels) + 1) + "|"]
        for name, res in robustness.items():
            lines.append(f"| {name} | " + " | ".join(f"{res[l]['accuracy']:.3f}" for l in levels) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
