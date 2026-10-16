"""Train the log-mel CNN and combine it with a feature-based Python model.

    pip install -r requirements-gtm.txt            # PyTorch (training only)
    python -m python_models.train_cnn              # base = the active Python model
    python -m python_models.train_cnn --base py-v4-ensemble-weighted --epochs 40 --no-activate

Pipeline
1. Log-mel spectrograms (``python_models/cnn.py``) of exactly the segments used
   for the feature model (``feature_extraction.build_features.segments_for_file``),
   cached in ``data/features/cnn/logmel.npz``.
2. The CNN is trained on the training split (originals + augmented copies) with
   SpecAugment masks; the epoch with the best validation macro F1 (recording
   level) is kept. The test split is not looked at while training.
3. The trained network is exported with batch norm folded into the
   convolutions and checked against PyTorch (``cnn.NumpyCNN``).
4. Hybrid: probabilities of the base feature model and the CNN are averaged
   with a weight chosen on the validation split, then critical-class decision
   weights are tuned on the validation split (same procedure as ``train.py``).
5. Candidates (base model, CNN alone, hybrid, hybrid + critical weighting) are
   ranked on validation macro F1 (critical recall breaks ties). A new model
   version is saved only if the winner is not the base model. The test split is
   used only for the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from config import DATASET_DIR, FEATURES_DIR, METADATA_CSV, MODELS_DIR, REPORTS_DIR

from .cnn import LOGMEL_CONFIG, NumpyCNN, fuse, logmel
from .ensemble import CriticalClassWeighting
from .evaluation import (
    CRITICAL_CLASSES,
    aggregate_by_recording,
    calibration_summary,
    classification_metrics,
    fit_temperature,
    plot_confusion_matrix,
    temper,
)
from .train import _jsonable, load_features, next_version, tune_critical_weights

CACHE = FEATURES_DIR / "cnn" / "logmel.npz"
TRAINED = FEATURES_DIR / "cnn" / "trained_cnn.joblib"  # trained network, reusable with another base model


# --------------------------------------------------------------------------- #
# Log-mel cache
# --------------------------------------------------------------------------- #
def _mels_for_row(row: dict, pre_cfg, seg_s: float, hop_s: float):
    from feature_extraction.build_features import segments_for_file

    try:
        segs, _ = segments_for_file(DATASET_DIR / row["relative_path"], row["class_label"], pre_cfg, seg_s, hop_s)
    except Exception as exc:  # a broken file must not stop the run (build_features skips it too)
        return [], [], f"{row['audio_id']}: {exc}"
    return [logmel(s.audio) for s in segs], [s.index for s in segs], None


def build_cache(jobs: int, classes: list[str] | None = None) -> dict:
    from joblib import Parallel, delayed

    from audio_preprocessing import PreprocessConfig

    # The exact settings features.npz was built with, so both members see the same segments.
    feat_cfg = json.loads((FEATURES_DIR / "feature_config.json").read_text(encoding="utf-8"))
    pre_cfg = PreprocessConfig(**feat_cfg["preprocessing"])
    seg_s, hop_s = feat_cfg["segment_seconds"], feat_cfg["segment_hop_seconds"]
    meta = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    if classes:
        meta = meta[meta["class_label"].isin(classes)]
    rows = meta.to_dict("records")  # same rows as feature_extraction/build_features.py
    fp = hashlib.sha256(json.dumps([feat_cfg, LOGMEL_CONFIG, sorted(r["audio_id"] for r in rows)],
                                   sort_keys=True, default=str).encode()).hexdigest()[:16]
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True)
        if str(data["fingerprint"]) == fp:
            print(f"Using cached log-mel spectrograms ({CACHE})")
            return {k: data[k] for k in data.files}
        if classes is not None:
            mask = np.isin(data["y"], classes)
            if mask.any():
                print(f"Using cached log-mels filtered to {len(classes)} classes ({mask.sum()} / {len(mask)} segments)")
                return {k: (data[k][mask] if (getattr(data[k], 'ndim', 0) > 0 and len(data[k]) == len(mask)) else data[k]) for k in data.files}
    print(f"Computing log-mel spectrograms for {len(rows)} recordings ...")
    t0 = time.perf_counter()
    out = Parallel(n_jobs=jobs, verbose=2, batch_size=8)(delayed(_mels_for_row)(r, pre_cfg, seg_s, hop_s) for r in rows)
    mels, meta_rows, errors = [], [], []
    for r, (m, idx, err) in zip(rows, out):
        if err:
            errors.append(err)
        for mel, i in zip(m, idx):
            mels.append(mel.astype(np.float16))
            meta_rows.append((r["audio_id"], r["root_audio_id"] or r["audio_id"], r["class_label"], r["split"],
                              r["status"], i))
    frames = {m.shape[1] for m in mels}
    if len(frames) != 1:
        raise RuntimeError(f"segments of different lengths: {sorted(frames)}")
    cols = list(zip(*meta_rows))
    data = {"X": np.stack(mels), "audio_id": np.array(cols[0]), "root_audio_id": np.array(cols[1]),
            "y": np.array(cols[2]), "split": np.array(cols[3]), "status": np.array(cols[4]),
            "segment_index": np.array(cols[5]), "fingerprint": np.array(fp)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **data)
    print(f"  {len(mels)} segments in {time.perf_counter() - t0:.0f} s, {len(errors)} errors")
    for e in errors[:10]:
        print("  !", e)
    return data


# --------------------------------------------------------------------------- #
# PyTorch model
# --------------------------------------------------------------------------- #
def build_torch_model(n_classes: int, channels=(16, 32, 64, 128), dropout: float = 0.3):
    import torch
    from torch import nn

    class LogMelCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c_in = [], 1
            for c in channels:
                layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                           nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)]
                c_in = c
            self.features = nn.Sequential(*layers)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(2 * c_in, n_classes)

        def forward(self, x):
            h = self.features(x)
            h = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
            return self.fc(self.dropout(h))

    return LogMelCNN()


def export_numpy(model, mean: float, std: float) -> dict:
    """Fold every BatchNorm into the convolution before it and return plain arrays."""
    from torch import nn

    convs, pending = [], None
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            pending = layer
        elif isinstance(layer, nn.BatchNorm2d):
            scale = (layer.weight / (layer.running_var + layer.eps).sqrt()).detach().cpu().numpy()
            w = pending.weight.detach().cpu().numpy() * scale[:, None, None, None]
            b = layer.bias.detach().cpu().numpy() - layer.running_mean.detach().cpu().numpy() * scale
            convs.append((w.astype(np.float32), b.astype(np.float32)))
    return {"config": dict(LOGMEL_CONFIG), "norm_mean": float(mean), "norm_std": float(std), "convs": convs,
            "convs_per_block": 2, "fc_w": model.fc.weight.detach().cpu().numpy().astype(np.float32),
            "fc_b": model.fc.bias.detach().cpu().numpy().astype(np.float32)}


def _spec_augment(x, rng: np.random.Generator):
    """Two frequency masks (<= 8 bands), two time masks (<= 12 frames) and a random time roll, per example."""
    x = x.clone()
    n, _, f, t = x.shape
    for i in range(n):
        for _ in range(2):
            w = int(rng.integers(0, 9))
            s = int(rng.integers(0, f - w + 1))
            x[i, :, s: s + w, :] = 0.0
            w = int(rng.integers(0, 13))
            s = int(rng.integers(0, t - w + 1))
            x[i, :, :, s: s + w] = 0.0
        x[i] = x[i].roll(int(rng.integers(-10, 11)), dims=2)
    return x


def train_cnn(Xn, y, tr, va, groups_va, classes, epochs: int, seed: int, threads: int):
    """Returns (model, history). Runs every epoch of the one-cycle schedule (stopping early would cut it off
    before the learning rate comes down) and keeps the epoch with the best validation macro F1 (recording
    level)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    model = build_torch_model(len(classes))
    counts = np.bincount(y[tr], minlength=len(classes)).astype(float)
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    steps_per_epoch = int(np.ceil(tr.sum() / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * steps_per_epoch)
    X_tr = torch.from_numpy(Xn[tr][:, None])
    y_tr = torch.from_numpy(y[tr].astype(np.int64))
    X_va = torch.from_numpy(Xn[va][:, None])
    best, best_state, history = (-1.0, -1.0), None, []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = torch.from_numpy(rng.permutation(len(X_tr)))
        total = 0.0
        for i in range(0, len(order), 64):
            idx = order[i: i + 64]
            xb = _spec_augment(X_tr[idx], rng)
            opt.zero_grad()
            loss = loss_fn(model(xb), y_tr[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            proba = torch.softmax(torch.cat([model(X_va[i: i + 256]) for i in range(0, len(X_va), 256)]), 1).numpy()
        _, agg, truth = aggregate_by_recording(proba, groups_va, y[va])
        m = classification_metrics(truth, agg.argmax(axis=1), classes)
        key = (round(m["f1_macro"], 4), m["critical_recall_min"])
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_accuracy": m["accuracy"],
                        "val_f1_macro": m["f1_macro"], "val_critical_recall_min": m["critical_recall_min"],
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  epoch {epoch:>2}: loss {total / len(order):.4f}  val acc {m['accuracy']:.4f}  "
              f"val macro F1 {m['f1_macro']:.4f}  crit {m['critical_recall_min']:.3f}  ({history[-1]['seconds']} s)")
        if key > best:
            best = key
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
class _Precomputed:
    """Wraps fixed probabilities so the validation-only tuning code of train.py can be reused."""

    def __init__(self, proba):
        self.proba = proba

    def predict_proba(self, _x):
        return self.proba


def _metrics(proba, groups, y, classes) -> dict:
    seg = classification_metrics(y, proba.argmax(axis=1), classes)
    _, agg, truth = aggregate_by_recording(proba, groups, y)
    return {"segment": seg, "recording": classification_metrics(truth, agg.argmax(axis=1), classes)}


def _noisy_inputs(row: dict, noise: np.ndarray, snr_db: float, pre_cfg, seg_s, hop_s, feat_cfg):
    """Features and log-mels of a test recording mixed with background noise (same mixing as train.py)."""
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
    segs = segs[:8]
    return ([extract_features(s.audio, feat_cfg) for s in segs], [logmel(s.audio) for s in segs],
            row["audio_id"], row["class_label"])


def noise_robustness(candidates: dict, classes, cfg, jobs: int, levels=(20, 10, 5, 0)) -> dict:
    """candidates: name -> function(features, mels) -> probabilities."""
    from joblib import Parallel, delayed

    from audio_preprocessing import PreprocessConfig
    from audio_preprocessing.loader import load_mono
    from feature_extraction.features import FeatureConfig

    meta = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    if classes:
        meta = meta[meta["class_label"].isin(classes)]
    test = meta[(meta["split"] == "test") & (meta["status"] == "original")].to_dict("records")
    bank_rows = meta[(meta["split"] == "train") & (meta["status"] == "original")
                     & (meta["class_label"] == "Background Noise")].head(40)
    pre_cfg = PreprocessConfig(**cfg["preprocessing"])
    feat_cfg = FeatureConfig.from_dict(cfg["feature_config"])
    bank = [load_mono(DATASET_DIR / p, pre_cfg.target_sr)[0] for p in bank_rows["relative_path"]]
    index = {c: i for i, c in enumerate(classes)}
    results = {name: {} for name in candidates}
    rng = np.random.default_rng(0)  # same noise picks as train.py
    for snr in levels:
        picks = [bank[int(rng.integers(0, len(bank)))] for _ in test]
        out = Parallel(n_jobs=jobs)(delayed(_noisy_inputs)(r, nz, snr, pre_cfg, cfg["segment_seconds"],
                                                            cfg["segment_hop_seconds"], feat_cfg)
                                    for r, nz in zip(test, picks))
        F, M, groups, y = [], [], [], []
        for feats, mels, aid, label in out:
            F += feats
            M += mels
            groups += [aid] * len(feats)
            y += [index[label]] * len(feats)
        F, groups, y = np.vstack(F), np.array(groups), np.array(y)
        for name, fn in candidates.items():
            _, agg, truth = aggregate_by_recording(fn(F, M), groups, y)
            m = classification_metrics(truth, agg.argmax(axis=1), classes)
            results[name][f"{snr} dB"] = {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                                          "critical_recall_min": m["critical_recall_min"]}
        print(f"  noise robustness @ {snr:>2} dB SNR: " + ", ".join(
            f"{n}={results[n][f'{snr} dB']['accuracy']:.3f}" for n in candidates))
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="feature-model version to combine with (default: the active model)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=6, help="processes for spectrograms and the noise test")
    parser.add_argument("--threads", type=int, default=6, help="PyTorch threads for training")
    parser.add_argument("--no-noise", action="store_true", help="skip the noise-robustness evaluation")
    parser.add_argument("--no-activate", action="store_true", help="save but do not make the model active")
    parser.add_argument("--reuse-cnn", action="store_true",
                        help="reuse the network trained by the last run (same data, seed and epochs)")
    parser.add_argument("--no-save", action="store_true", help="only print the comparison, save no model version")
    args = parser.parse_args(argv)

    base_version = args.base or json.loads((MODELS_DIR / "active.json").read_text(encoding="utf-8"))["version"]
    base_art = joblib.load(MODELS_DIR / f"{base_version}.joblib")
    base_card = base_art["card"]
    if base_art.get("cnn"):
        raise SystemExit(f"{base_version} already contains a CNN; choose a feature-only model with --base.")
    base_model = base_art["model"]
    feature_model = base_model.base if isinstance(base_model, CriticalClassWeighting) else base_model
    classes = list(base_card["classes"])
    index = {c: i for i, c in enumerate(classes)}
    print(f"Base feature model: {base_version} ({base_card['algorithm_label']})")

    mel = build_cache(args.jobs, classes)
    feats, cfg = load_features()
    y_m = np.array([index[c] for c in mel["y"]])
    split, status, aid = mel["split"], mel["status"], mel["audio_id"]
    tr = split == "train"
    va = (split == "validation") & (status == "original")
    te = (split == "test") & (status == "original")
    roots = mel["root_audio_id"]
    assert not (set(roots[tr]) & set(roots[va])) and not (set(roots[tr]) & set(roots[te])), "split leakage"

    # Align the feature rows with the spectrogram rows (same recording + segment index).
    # The spectrogram cache holds the upload rendering only, so only those feature rows may be matched.
    f_rendering = feats.get("rendering")
    if f_rendering is None:
        f_rendering = np.full(len(feats["audio_id"]), "upload")
    upload = np.flatnonzero(f_rendering == "upload")
    key_f = {(feats["audio_id"][i], int(feats["segment_index"][i])): int(i) for i in upload}
    rows_f = np.array([key_f.get((a, int(s)), -1) for a, s in zip(aid, mel["segment_index"])])
    if (rows_f[va | te] < 0).any():
        raise SystemExit("validation/test segments of the spectrogram cache and features.npz differ; "
                         "rebuild features (python -m feature_extraction.build_features)")

    X32 = mel["X"].astype(np.float32)
    mean, std = float(X32[tr].mean()), float(X32[tr].std())
    Xn = (X32 - mean) / std
    print(f"Segments: train {tr.sum()}, validation {va.sum()}, test {te.sum()}; log-mel {X32.shape[1:]}")

    run_key = [str(mel["fingerprint"]), args.seed, args.epochs]
    saved = joblib.load(TRAINED) if args.reuse_cnn and TRAINED.exists() else None
    if saved is not None and saved.get("run_key") == run_key:
        print(f"Reusing the trained CNN from {TRAINED}")
        export, history, fit_seconds, check = saved["export"], saved["history"], saved["fit_seconds"], saved["check"]
        cnn = NumpyCNN(export)
    else:
        if args.reuse_cnn:
            print("No matching trained CNN to reuse; training a new one.")
        t0 = time.perf_counter()
        torch_model, history = train_cnn(Xn, y_m, tr, va, aid[va], classes, args.epochs, args.seed, args.threads)
        fit_seconds = time.perf_counter() - t0
        export = export_numpy(torch_model, mean, std)
        cnn = NumpyCNN(export)

        import torch

        with torch.no_grad():
            ref = torch.softmax(torch_model(torch.from_numpy(Xn[va][:128, None])), 1).numpy()
        check = float(np.abs(cnn.predict_proba_mels(list(X32[va][:128])) - ref).max())
        print(f"Export check: max |NumPy - PyTorch| probability difference = {check:.2e}")
        if check > 1e-3:
            raise SystemExit("exported network does not match the trained one")
        joblib.dump({"run_key": run_key, "export": export, "history": history, "fit_seconds": fit_seconds,
                     "check": check}, TRAINED, compress=3)

    def cnn_proba(mask):
        t = time.perf_counter()
        p = cnn.predict_proba_mels(list(X32[mask]), batch=64)
        return p, (time.perf_counter() - t) / max(1, mask.sum()) * 1000

    p_cnn_va, _ = cnn_proba(va)
    p_cnn_te, cnn_ms = cnn_proba(te)
    p_feat_va = np.asarray(feature_model.predict_proba(feats["X"][rows_f[va]]), float)
    p_feat_te = np.asarray(feature_model.predict_proba(feats["X"][rows_f[te]]), float)
    p_base_va = np.asarray(base_model.predict_proba(feats["X"][rows_f[va]]), float)
    p_base_te = np.asarray(base_model.predict_proba(feats["X"][rows_f[te]]), float)
    y_va, y_te, g_va, g_te = y_m[va], y_m[te], aid[va], aid[te]

    # Fusion weight on validation.
    grid = [round(w, 2) for w in np.arange(0.0, 1.0001, 0.05)]
    scored = []
    for w in grid:
        m = _metrics(fuse(p_feat_va, p_cnn_va, w), g_va, y_va, classes)["recording"]
        scored.append(((round(m["f1_macro"], 3), m["critical_recall_min"]), w))
    cnn_weight = max(scored)[1]
    p_hyb_va = fuse(p_feat_va, p_cnn_va, cnn_weight)
    p_hyb_te = fuse(p_feat_te, p_cnn_te, cnn_weight)
    crit_w, crit_info = tune_critical_weights(_Precomputed(p_hyb_va), None, y_va, g_va, classes)
    print(f"Fusion weight (validation): CNN {cnn_weight:.2f}, features {1 - cnn_weight:.2f}; "
          f"critical weights {dict((classes[i], round(float(w), 2)) for i, w in enumerate(crit_w) if w != 1)}")

    cnn_params = {"epochs_run": len(history), "best_epoch": int(np.argmax([h["val_f1_macro"] for h in history]) + 1),
                  "optimizer": "AdamW lr 2e-3, weight decay 1e-3, one-cycle schedule (max 3e-3)", "batch": 64,
                  "spec_augment": "2 frequency masks <= 8 bands, 2 time masks <= 12 frames, time roll +-10 frames",
                  "label_smoothing": 0.05, "class_weights": "inverse class frequency", "seed": args.seed}
    base_label = base_card["algorithm_label"]
    candidates = {
        "base": (base_label + f" [{base_version}]", p_base_va, p_base_te, None),
        "cnn": ("Log-mel CNN (4 conv blocks, PyTorch-trained, NumPy inference)", p_cnn_va, p_cnn_te, None),
        "hybrid": (f"Hybrid: {base_label.split(' + critical')[0]} + log-mel CNN (weight {cnn_weight:.2f})",
                   p_hyb_va, p_hyb_te, None),
    }
    if not np.allclose(crit_w, 1.0):
        candidates["hybrid_weighted"] = (candidates["hybrid"][0] + " + critical-class weighting",
                                         fuse(p_feat_va, p_cnn_va, cnn_weight, crit_w),
                                         fuse(p_feat_te, p_cnn_te, cnn_weight, crit_w), crit_w)
    results = {n: {"label": lab, "validation": _metrics(pv, g_va, y_va, classes),
                   "test": _metrics(pt, g_te, y_te, classes)} for n, (lab, pv, pt, _) in candidates.items()}
    for n, r in results.items():
        v, t = r["validation"]["recording"], r["test"]["recording"]
        print(f"  {n:<16} val acc {v['accuracy']:.4f} F1 {v['f1_macro']:.4f} crit {v['critical_recall_min']:.3f} | "
              f"test acc {t['accuracy']:.4f} F1 {t['f1_macro']:.4f} crit {t['critical_recall_min']:.3f}")

    def rank_key(n):
        v = results[n]["validation"]["recording"]
        return (round(v["f1_macro"], 3), v["critical_recall_min"])

    best = max(results, key=rank_key)
    if args.no_save:
        print(f"Best on validation: {results[best]['label']} (--no-save: nothing written)")
        return 0
    print(f"\nSelected on validation: {results[best]['label']}")
    if best == "base":
        print("The CNN did not improve the base model on the validation split; no new version saved.")
        return 0
    if best == "cnn":
        cnn_weight, crit_w = 1.0, np.ones(len(classes))
    elif best == "hybrid":
        crit_w = np.ones(len(classes))
    # Confidence calibration of the selected combination (validation segments, log loss).
    p_best_va, p_best_te = candidates[best][1], candidates[best][2]
    temperature = fit_temperature(p_best_va, y_va)
    _, r_te, r_y = aggregate_by_recording(p_best_te, g_te, y_te)
    _, r_te_cal, _ = aggregate_by_recording(temper(p_best_te, temperature), g_te, y_te)
    calibration = {
        "method": "temperature scaling of the fused class probabilities, fitted on the validation segments",
        "temperature": round(temperature, 4),
        "validation_segments": {"before": calibration_summary(p_best_va, y_va),
                                "after": calibration_summary(temper(p_best_va, temperature), y_va)},
        "test_segments": {"before": calibration_summary(p_best_te, y_te),
                          "after": calibration_summary(temper(p_best_te, temperature), y_te)},
        "test_recordings": {"before": calibration_summary(r_te, r_y), "after": calibration_summary(r_te_cal, r_y)},
    }
    print(f"Calibration: temperature {temperature:.3f}; test segment ECE "
          f"{calibration['test_segments']['before']['ece']:.3f} -> {calibration['test_segments']['after']['ece']:.3f}")

    robustness = {}
    if not args.no_noise:
        print("\nNoise-robustness evaluation on the test split ...")
        fns = {
            "base": lambda F, M: base_model.predict_proba(F),
            "cnn": lambda F, M: cnn.predict_proba_mels(M, batch=64),
            best: lambda F, M: fuse(feature_model.predict_proba(F), cnn.predict_proba_mels(M, batch=64),
                                    cnn_weight, crit_w),
        }
        robustness = noise_robustness(fns, classes, cfg, args.jobs)

    version = next_version(best.replace("hybrid", "hybrid-cnn"))
    report_dir = REPORTS_DIR / "python_model" / version
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = [dict(r, model=f"{r['model']} ({base_version})", selected=False) for r in base_card["comparison"]]
    for n, r in results.items():
        if n == "base":
            continue
        v, t = r["validation"]["recording"], r["test"]["recording"]
        comparison.append({
            "model": n, "label": r["label"], "selected": n == best, "cv_f1_macro": float("nan"),
            "val_accuracy": round(v["accuracy"], 4), "val_f1_macro": round(v["f1_macro"], 4),
            "val_critical_recall_min": round(v["critical_recall_min"], 4),
            "test_accuracy": round(t["accuracy"], 4), "test_precision_macro": round(t["precision_macro"], 4),
            "test_recall_macro": round(t["recall_macro"], 4), "test_f1_macro": round(t["f1_macro"], 4),
            "test_critical_recall_min": round(t["critical_recall_min"], 4),
            "test_segment_accuracy": round(r["test"]["segment"]["accuracy"], 4),
            "noise_0db_accuracy": round(robustness.get(n, {}).get("0 dB", {}).get("accuracy", float("nan")), 4),
            "tune_seconds": 0.0, "fit_seconds": round(fit_seconds, 1) if n == "cnn" else 0.0,
            "inference_ms_per_segment": round(cnn_ms, 3),
            "hyperparameters": json.dumps(cnn_params if n == "cnn" else
                                          {"cnn_weight": cnn_weight, "base": base_version,
                                           "critical_weights": n == "hybrid_weighted"}),
        })
    for n, r in results.items():
        plot_confusion_matrix(r["test"]["recording"]["confusion_matrix"], classes,
                              f"{r['label']} - test set (recording level)", report_dir / f"confusion_matrix_{n}.png")
    pd.DataFrame(comparison).to_csv(report_dir / "model_comparison.csv", index=False)
    pd.DataFrame(history).to_csv(report_dir / "training_history.csv", index=False)

    ids, agg, truth = aggregate_by_recording(temper(p_best_te, temperature), g_te, y_te)
    samples = []
    for c in range(len(classes)):
        for i in np.flatnonzero(truth == c)[:3]:
            top = np.argsort(agg[i])[::-1][:3]
            samples.append({"audio_id": ids[i], "actual": classes[truth[i]], "predicted": classes[top[0]],
                            "confidence": round(float(agg[i][top[0]]), 4),
                            "top3": "; ".join(f"{classes[j]} {agg[i][j]:.2f}" for j in top),
                            "correct": bool(top[0] == truth[i])})
    pd.DataFrame(samples).to_csv(report_dir / "sample_predictions.csv", index=False)

    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    card = dict(base_card)
    card.update({
        "version": version,
        "algorithm": best,
        "algorithm_label": results[best]["label"],
        "trained_at": trained_at,
        "critical_classes": CRITICAL_CLASSES,
        "base_version": base_version,
        "hyperparameters": {
            "feature_model": base_card["hyperparameters"],
            "cnn": {"architecture": "4 x [conv3x3-BN-ReLU x2, maxpool2], channels 16/32/64/128, "
                                    f"global mean+max pooling, dropout 0.3, dense {len(classes)}",
                    "input": LOGMEL_CONFIG, "epochs_run": len(history), "seed": args.seed},
            "fusion": {"cnn_weight": cnn_weight, "tuned_on": "validation split",
                       "critical_weights": {classes[i]: round(float(w), 2) for i, w in enumerate(crit_w) if w != 1},
                       "critical_weighting_effect": crit_info},
        },
        "metrics": {"validation": results[best]["validation"], "test": results[best]["test"],
                    "noise_robustness": robustness.get(best, {})},
        "calibration": calibration,
        "comparison": comparison,
        "noise_robustness_all": robustness,
        "export_check_max_abs_diff": check,
        "report_dir": str(report_dir.relative_to(REPORTS_DIR.parent)).replace("\\", "/"),
        "file": f"{version}.joblib",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    })
    card = _jsonable(card)
    artifact = {"model": feature_model if best != "cnn" else base_model, "card": card, "cnn": export,
                "fusion": {"cnn_weight": float(cnn_weight), "critical_weights": [float(w) for w in crit_w]},
                "temperature": temperature}
    joblib.dump(artifact, MODELS_DIR / f"{version}.joblib", compress=3)
    (MODELS_DIR / f"{version}.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(_jsonable(results), indent=2, default=str), encoding="utf-8")
    (report_dir / "noise_robustness.json").write_text(json.dumps(_jsonable(robustness), indent=2), encoding="utf-8")
    write_summary(report_dir / "summary.md", card, results, best, robustness, history, check)
    if not args.no_activate:
        (MODELS_DIR / "active.json").write_text(json.dumps({"version": version, "file": f"{version}.joblib",
                                                             "activated_at": trained_at}, indent=2), encoding="utf-8")
    print(f"\nSaved {MODELS_DIR / (version + '.joblib')}{'' if args.no_activate else ' (active)'}\n"
          f"Reports in {report_dir}")
    return 0


def write_summary(path, card, results, best, robustness, history, check) -> None:
    t = card["metrics"]["test"]["recording"]
    lines = [f"# Python model {card['version']}", "", f"{card['algorithm_label']}", "",
             f"Base feature model: `{card['base_version']}`. Trained {card['trained_at']}.", "",
             "## Candidates (recording level; selection on validation macro F1)", "",
             "| Candidate | Val acc. | Val macro F1 | Val min critical recall | Test acc. | Test macro F1 | "
             "Test min critical recall | Selected |", "|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        v, te = r["validation"]["recording"], r["test"]["recording"]
        lines.append(f"| {r['label']} | {v['accuracy']:.3f} | {v['f1_macro']:.3f} | {v['critical_recall_min']:.3f} | "
                     f"{te['accuracy']:.3f} | {te['f1_macro']:.3f} | {te['critical_recall_min']:.3f} | "
                     f"{'yes' if n == best else ''} |")
    lines += ["", "## Selected model — test set, class-wise", "", "| Class | Precision | Recall | F1 |",
              "|---|---|---|---|"]
    for c, m in t["per_class"].items():
        lines.append(f"| {c} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    lines += ["", f"Test: accuracy {t['accuracy']:.3f}, macro precision {t['precision_macro']:.3f}, "
              f"macro recall {t['recall_macro']:.3f}, macro F1 {t['f1_macro']:.3f}, "
              f"min critical recall {t['critical_recall_min']:.3f} ({t.get('n', '')} recordings).", ""]
    if robustness:
        lines += ["## Noise robustness (test split + background noise)", "", "| Model | " +
                  " | ".join(next(iter(robustness.values())).keys()) + " |",
                  "|---|" + "---|" * len(next(iter(robustness.values())))]
        for n, r in robustness.items():
            lines.append(f"| {n} | " + " | ".join(f"{v['accuracy']:.3f}" for v in r.values()) + " |")
        lines.append("")
    best_epoch = int(np.argmax([h["val_f1_macro"] for h in history]) + 1)
    lines += [f"CNN training: {len(history)} epochs, best validation macro F1 at epoch {best_epoch} "
              f"(`training_history.csv`). Export check: max |NumPy − PyTorch| = {check:.1e}.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
