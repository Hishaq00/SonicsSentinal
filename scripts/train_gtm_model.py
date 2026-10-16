"""Train the Google Teachable Machine (GTM) audio model on the project dataset.

    python -m scripts.train_gtm_model               # full run (about 30-60 minutes on a laptop CPU)
    python -m scripts.train_gtm_model --quick       # short run to check the pipeline
    python -m scripts.train_gtm_model --no-export   # train and report, keep gtm_model/model/ as it is

Training needs PyTorch. The web application does not: it runs the exported
model in the browser with TensorFlow.js.
    python -m pip install -r requirements-gtm.txt

Trained exactly like the Python model
* Same recordings and splits: fitted on the train split (original recordings
  and their augmented copies), selected on the validation split, reported on
  the unseen test split (original recordings only).
* Same audio path: every recording goes through the web application's own
  pre-processing and 2 s / 1 s segmentation, keeping the event segments (every
  segment for Background Noise), at most 8 per recording - the rule of
  feature_extraction/build_features.py.
* GTM's input for a segment is exactly what static/js/gtm.js computes in the
  browser: processed.wav (16-bit) at 44.1 kHz, the loudest ~1 s window of the
  segment, 43 x 232 spectrogram, z-score. gtm_model/pipeline.py reproduces it
  bit for bit, dither included.
* Same evaluation and selection: recording level (mean of the segment
  probabilities), accuracy, macro precision / recall / F1, confusion matrix and
  critical-class recall (python_models/evaluation.py). The best validation
  macro F1 wins, critical recall breaks ties. The same critical-class weighting
  search runs on the validation split; the weights are folded into the output
  layer's bias, so the browser applies them too.

The network is the Teachable Machine audio network
Teachable Machine audio projects put a new softmax layer ("NewHeadDense") on
Google's speech-commands BROWSER_FFT model (4 convolution blocks and a
2000-unit dense layer) and train it on the recorded examples. We start from the
same pretrained base and train two candidates:
  tm-transfer   base frozen, only the new layer trained - Teachable Machine's
                own procedure (50 epochs, batch size 16, learning rate 0.001);
  tm-finetuned  the whole network fine-tuned on this dataset, with time-shift
                and SpecAugment augmentation, on both the upload rendering and
                the live-microphone rendering of the training recordings.
The exported files have the topology of a Teachable Machine audio export and
are checked with an independent NumPy implementation of the layers
(gtm_model/inference.py) before they replace gtm_model/model/.

Outputs
    gtm_model/model/            model.json, weights.bin, metadata.json
    reports/gtm_model/<version>/ summary.md, metrics.json, model_comparison.csv,
                                 confusion_matrix_*.png, training_history.csv,
                                 sample_predictions.csv
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    DATASET_DIR,
    FEATURES_DIR,
    GTM_DIR,
    GTM_MODEL_DIR,
    METADATA_CSV,
    REPORTS_DIR,
    class_names,
    load_default_settings,
)
from feature_extraction.gtm_spectrogram import GTM_BINS, GTM_EPSILON, GTM_FFT, GTM_FRAMES, GTM_HOP  # noqa: E402
from gtm_model import pipeline  # noqa: E402
from gtm_model.inference import TFJSLayersModel, read_weights  # noqa: E402
from python_models.evaluation import (  # noqa: E402
    CRITICAL_CLASSES,
    aggregate_by_recording,
    calibration_summary,
    classification_metrics,
    fit_temperature,
    plot_confusion_matrix,
)

BASE_URL = "https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/"
BASE_FILES = ("model.json", "metadata.json", "group1-shard1of2", "group1-shard2of2")
BASE_DIR = GTM_DIR / "base_model" / "speech-commands-18w"
BASE_LAYERS = ("conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "dense_1")
HEAD = "NewHeadDense"  # the speech-commands / Teachable Machine name of the new output layer
CACHE_DIR = FEATURES_DIR / "gtm"
CACHE_VERSION = 1
EXTRA_FRAMES = 10  # time-shift augmentation: up to +-10 frames (+-0.23 s) around the browser's window


# --------------------------------------------------------------------------- #
# Data: GTM inputs exactly as the browser computes them
# --------------------------------------------------------------------------- #
def _extract(row: dict, path_kind: str, context: bool, settings: dict):
    try:
        audio = pipeline.load(DATASET_DIR / row["relative_path"], path_kind, settings)
        segs = pipeline.training_segments(audio, row["class_label"])
        if context:
            arrays = [pipeline.context_spectrogram(audio, s, EXTRA_FRAMES).astype(np.float16) for s in segs]
        else:
            arrays = [pipeline.segment_input(audio, s)[0] for s in segs]
        return arrays, [s.index for s in segs], None
    except Exception as exc:  # a broken file must not stop the whole run
        return [], [], f"{row['audio_id']}: {exc}"


def _build_set(rows: list[dict], path_kind: str, context: bool, settings: dict, jobs: int):
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=jobs, batch_size=8)(delayed(_extract)(r, path_kind, context, settings) for r in rows)
    arrays, meta, errors = [], [], []
    for row, (arrs, indices, err) in zip(rows, results):
        if err:
            errors.append(err)
        for a, i in zip(arrs, indices):
            arrays.append(a)
            meta.append({"audio_id": row["audio_id"], "root_audio_id": row["root_audio_id"] or row["audio_id"],
                         "class_label": row["class_label"], "split": row["split"], "status": row["status"],
                         "segment_index": i})
    return np.stack(arrays), pd.DataFrame(meta), errors


def _fingerprint(meta: pd.DataFrame, settings: dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([CACHE_VERSION, EXTRA_FRAMES, GTM_FRAMES, GTM_BINS, GTM_FFT, GTM_HOP,
                         settings["preprocessing"], settings["segmentation"]], sort_keys=True).encode())
    h.update("".join(sorted(meta["audio_id"] + meta["split"] + meta["sha256"] + meta["relative_path"])).encode())
    return h.hexdigest()[:16]


def load_sets(meta: pd.DataFrame, settings: dict, names: list[str], jobs: int, force: bool) -> dict:
    """{name: (array, segment metadata)}; name = '<split>-<upload|live>'. Cached in data/features/gtm/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(meta, settings)
    info_path = CACHE_DIR / "cache.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    if force or info.get("fingerprint") != fp:
        info = {"fingerprint": fp, "sets": {}}
    out = {}
    for name in names:
        split, path_kind = name.split("-")
        npy, csv = CACHE_DIR / f"{name}.npy", CACHE_DIR / f"{name}.csv"
        if not (name in info["sets"] and npy.exists() and csv.exists()):
            rows = meta[(meta["split"] == split) & ((meta["status"] == "original") | (split == "train"))]
            t0 = time.time()
            X, m, errors = _build_set(rows.to_dict("records"), path_kind, split == "train", settings, jobs)
            np.save(npy, X)
            m.to_csv(csv, index=False)
            info["sets"][name] = {"segments": int(len(X)), "recordings": int(m["audio_id"].nunique()), "errors": errors}
            info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(f"  {name:18s} {len(X):6d} windows from {m['audio_id'].nunique():5d} recordings "
                  f"({time.time() - t0:.0f} s){f', {len(errors)} errors' if errors else ''}", flush=True)
        else:
            print(f"  {name:18s} cached ({info['sets'][name]['segments']} windows)", flush=True)
        m = pd.read_csv(csv, dtype=str, keep_default_na=False)
        m["segment_index"] = m["segment_index"].astype(int)
        out[name] = (np.load(npy), m)
    return out


# --------------------------------------------------------------------------- #
# Speech-commands base model (the network Teachable Machine audio builds on)
# --------------------------------------------------------------------------- #
def ensure_base_model() -> dict:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for name in BASE_FILES:
        target = BASE_DIR / name
        if not target.exists() or target.stat().st_size == 0:
            print(f"  downloading {BASE_URL}{name}", flush=True)
            with urllib.request.urlopen(BASE_URL + name, timeout=120) as resp:
                target.write_bytes(resp.read())
    model_json = json.loads((BASE_DIR / "model.json").read_text(encoding="utf-8"))
    return {"model_json": model_json, "weights": read_weights(BASE_DIR, model_json["weightsManifest"])}


def build_network(n_classes: int):
    from torch import nn
    from torch.nn import functional as F

    class TMAudioNet(nn.Module):
        """Teachable Machine audio network: speech-commands 18w base + a new softmax layer."""

        def __init__(self):
            super().__init__()
            self.conv2d_1 = nn.Conv2d(1, 8, (2, 8))
            self.conv2d_2 = nn.Conv2d(8, 32, (2, 4))
            self.conv2d_3 = nn.Conv2d(32, 32, (2, 4))
            self.conv2d_4 = nn.Conv2d(32, 32, (2, 4))
            self.dropout_1 = nn.Dropout(0.25)
            self.dense_1 = nn.Linear(704, 2000)
            self.dropout_2 = nn.Dropout(0.5)
            self.head = nn.Linear(2000, n_classes)

        def embed(self, x):
            x = x.unsqueeze(1)  # (N, 1, 43, 232)
            x = F.max_pool2d(F.relu(self.conv2d_1(x)), (2, 2), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2d_2(x)), (2, 2), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2d_3(x)), (2, 2), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2d_4(x)), (2, 2), (1, 2))
            x = x.permute(0, 2, 3, 1).flatten(1)  # TensorFlow.js flattens channels_last
            return F.relu(self.dense_1(self.dropout_1(x)))

        def forward(self, x):
            return self.head(self.dropout_2(self.embed(x)))

    return TMAudioNet()


def load_base_weights(net, weights: dict[str, np.ndarray], head: str | None = None) -> None:
    """TensorFlow.js kernels (kh, kw, in, out) / (in, out) -> PyTorch layout."""
    import torch

    names = list(BASE_LAYERS) + ([head] if head else [])
    with torch.no_grad():
        for name in names:
            kernel, bias = weights[f"{name}/kernel"], weights[f"{name}/bias"]
            layer = net.head if name == head else getattr(net, name)
            w = kernel.transpose(3, 2, 0, 1) if kernel.ndim == 4 else kernel.T
            layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)))
            layer.bias.copy_(torch.from_numpy(np.ascontiguousarray(bias)))


def export_tfjs(net, labels: list[str], base_json: dict, out_dir: Path, generated_by: str) -> None:
    """Write model.json + weights.bin with the Teachable Machine audio topology."""
    layers = []
    for layer in base_json["modelTopology"]["model_config"]["config"]:
        if layer["config"]["name"] == "dense_2":
            head = copy.deepcopy(layer)
            head["config"].update(name=HEAD, units=len(labels))
            layers.append(head)
            break
        layers.append(copy.deepcopy(layer))
    specs, blobs = [], []

    def add(name: str, arr) -> None:
        arr = np.ascontiguousarray(arr, dtype="<f4")
        specs.append({"name": name, "shape": list(arr.shape), "dtype": "float32"})
        blobs.append(arr.tobytes())

    for name in (*BASE_LAYERS, HEAD):
        layer = net.head if name == HEAD else getattr(net, name)
        w = layer.weight.detach().cpu().numpy()
        add(f"{name}/kernel", w.transpose(2, 3, 1, 0) if w.ndim == 4 else w.T)
        add(f"{name}/bias", layer.bias.detach().cpu().numpy())
    topology = {"class_name": "Sequential", "config": {"name": "sonicsentinel_gtm", "layers": layers},
                "keras_version": "tfjs-layers 4.22.0", "backend": "tensor_flow.js"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(b"".join(blobs))
    (out_dir / "model.json").write_text(json.dumps({
        "format": "layers-model", "generatedBy": generated_by, "convertedBy": None,
        "modelTopology": topology, "weightsManifest": [{"paths": ["weights.bin"], "weights": specs}],
    }, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def normalise(x: np.ndarray) -> np.ndarray:
    """speech-commands normalize() per example: (x - mean) / (std + epsilon)."""
    m = x.mean(axis=(1, 2), keepdims=True, dtype=np.float64)
    s = x.std(axis=(1, 2), keepdims=True, dtype=np.float64)
    return ((x - m) / (s + GTM_EPSILON)).astype(np.float32)


def crops(ctx: np.ndarray, idx: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    frames = offsets[:, None] + np.arange(GTM_FRAMES)[None, :]
    return normalise(np.asarray(ctx[idx[:, None], frames], dtype=np.float32))


def spec_augment(x: np.ndarray, rng: np.random.Generator, max_bins: int = 16, max_frames: int = 5,
                 p: float = 0.5) -> np.ndarray:
    """One frequency mask and one time mask per example (each with probability ``p``)."""
    n = len(x)
    fw = rng.integers(0, max_bins + 1, n) * (rng.random(n) < p)
    f0 = rng.integers(0, GTM_BINS - fw + 1)
    tw = rng.integers(0, max_frames + 1, n) * (rng.random(n) < p)
    t0 = rng.integers(0, GTM_FRAMES - tw + 1)
    b, t = np.arange(GTM_BINS)[None, :], np.arange(GTM_FRAMES)[None, :]
    fmask = (b >= f0[:, None]) & (b < (f0 + fw)[:, None])
    tmask = (t >= t0[:, None]) & (t < (t0 + tw)[:, None])
    x[fmask[:, None, :] | tmask[:, :, None]] = 0.0
    return x


def predict(net, X: np.ndarray, batch: int = 256) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch], dtype=np.float32))
            out.append(torch.softmax(net(xb), dim=1).numpy())
    return np.vstack(out)


def evaluate(proba: np.ndarray, meta: pd.DataFrame, classes: list[str]) -> dict:
    index = {c: i for i, c in enumerate(classes)}
    y = meta["class_label"].map(index).to_numpy()
    seg = classification_metrics(y, proba.argmax(axis=1), classes)
    _, agg, truth = aggregate_by_recording(proba, meta["audio_id"].to_numpy(), y)
    return {"segment": seg, "recording": classification_metrics(truth, agg.argmax(axis=1), classes)}


def rank_key(ev: dict) -> tuple:
    """The Python model's selection rule: validation macro F1, then critical-class recall."""
    r = ev["recording"]
    return round(r["f1_macro"], 3), r["critical_recall_min"]


def apply_class_weights(proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """What adding log(weights) to the output bias does to the softmax."""
    p = proba * weights
    return p / p.sum(axis=1, keepdims=True)


def tune_critical_weights(proba: np.ndarray, meta: pd.DataFrame, classes: list[str], max_f1_drop: float = 0.01):
    """python_models.train.tune_critical_weights, for weights applied inside the model's softmax."""
    critical = [i for i, c in enumerate(classes) if c in CRITICAL_CLASSES]

    def score(w):
        r = evaluate(apply_class_weights(proba, w), meta, classes)["recording"]
        return r["critical_recall_min"], r["f1_macro"]

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


def _class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return (len(y) / (n_classes * np.maximum(counts, 1))).astype(np.float32)


def train_transfer(net, ctx: np.ndarray, y: np.ndarray, val_X: np.ndarray, val_meta: pd.DataFrame,
                   classes: list[str], epochs: int, seed: int, history: list) -> dict:
    """Teachable Machine's training: base frozen, only the new softmax layer learns
    (batch size 16, Adam, learning rate 0.001) on the windows the browser classifies."""
    import torch
    from torch.nn import functional as F

    rng = np.random.default_rng(seed)
    net.eval()
    with torch.no_grad():
        centre = np.full(len(ctx), EXTRA_FRAMES)
        emb = np.vstack([net.embed(torch.from_numpy(crops(ctx, np.arange(i, min(i + 256, len(ctx))),
                                                          centre[i:i + 256]))).numpy()
                         for i in range(0, len(ctx), 256)])
        val_emb = torch.cat([net.embed(torch.from_numpy(np.ascontiguousarray(val_X[i:i + 256])))
                             for i in range(0, len(val_X), 256)])
    head = net.head
    torch.nn.init.xavier_uniform_(head.weight)  # Keras / TensorFlow.js default (glorot uniform)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    cw = torch.from_numpy(_class_weights(y, len(classes)))
    emb_t, y_t = torch.from_numpy(emb), torch.from_numpy(y.astype(np.int64))
    best, best_state = None, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(order), 16):
            b = torch.from_numpy(order[i:i + 16])
            loss = F.cross_entropy(head(emb_t[b]), y_t[b], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        with torch.no_grad():
            ev = evaluate(torch.softmax(head(val_emb), dim=1).numpy(), val_meta, classes)
        _log(history, "tm-transfer", epoch, total / len(y), ev, 1e-3, time.time() - t0)
        if best is None or rank_key(ev) > best:
            best, best_state = rank_key(ev), copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return copy.deepcopy(net.state_dict())


def train_finetune(net, pools: list[np.ndarray], labels: list[np.ndarray], val_X: np.ndarray,
                   val_meta: pd.DataFrame, classes: list[str], epochs: int, patience: int, lr: float,
                   seed: int, history: list, ckpt: dict | None = None, ckpt_path: Path | None = None) -> dict:
    """Fine-tune the whole network with time-shift + SpecAugment augmentation.

    With ``ckpt_path`` the training state is saved after every epoch; a ``ckpt``
    that already holds a "finetune" entry continues where that run stopped.
    """
    import torch
    from torch.nn import functional as F

    rng = np.random.default_rng(seed + 1)
    pool_id = np.concatenate([np.full(len(p), k) for k, p in enumerate(pools)])
    row_id = np.concatenate([np.arange(len(p)) for p in pools])
    y_all = np.concatenate(labels)
    cw = torch.from_numpy(_class_weights(y_all, len(classes)))
    batch = 64
    steps = epochs * int(np.ceil(len(y_all) / batch))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos", final_div_factor=20.0)
    best, best_state, since, start, done = None, None, 0, 1, False
    state = (ckpt or {}).get("finetune")
    if state:
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        rng.bit_generator.state = state["rng"]
        best, best_state, since, done = tuple(state["best"]), state["best_state"], state["since"], state["done"]
        start = state["epoch"] + 1
        print(f"    resuming after epoch {state['epoch']} (best validation macro F1 {best[0]:.3f})", flush=True)
    for epoch in range(start, epochs + 1):
        if done:
            break
        t0 = time.time()
        net.train()
        order = rng.permutation(len(y_all))
        total = 0.0
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            offsets = rng.integers(0, 2 * EXTRA_FRAMES + 1, len(b))
            x = np.empty((len(b), GTM_FRAMES, GTM_BINS), dtype=np.float32)
            for k, pool in enumerate(pools):
                sel = pool_id[b] == k
                if sel.any():
                    x[sel] = crops(pool, row_id[b][sel], offsets[sel])
            x = spec_augment(x, rng)
            logits = net(torch.from_numpy(x))
            loss = F.cross_entropy(logits, torch.from_numpy(y_all[b].astype(np.int64)), weight=cw,
                                   label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched.last_epoch + 1 < steps:
                sched.step()
            total += loss.item() * len(b)
        ev = evaluate(predict(net, val_X), val_meta, classes)
        _log(history, "tm-finetuned", epoch, total / len(y_all), ev, opt.param_groups[0]["lr"], time.time() - t0)
        if best is None or rank_key(ev) > best:
            best, best_state, since = rank_key(ev), copy.deepcopy(net.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                print(f"    early stop: no validation improvement for {patience} epochs", flush=True)
                done = True
        if ckpt_path is not None:
            ckpt["finetune"] = {"epoch": epoch, "net": net.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "rng": rng.bit_generator.state, "best": best,
                                "best_state": best_state, "since": since, "done": done or epoch == epochs}
            ckpt["history"] = history
            save_checkpoint(ckpt, ckpt_path)
    net.load_state_dict(best_state)
    return best_state


def save_checkpoint(ckpt: dict, path: Path) -> None:
    """Write atomically, so an interrupted save never leaves a broken checkpoint."""
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _log(history: list, phase: str, epoch: int, loss: float, ev: dict, lr: float, seconds: float) -> None:
    r = ev["recording"]
    history.append({"phase": phase, "epoch": epoch, "train_loss": round(loss, 4),
                    "val_accuracy": round(r["accuracy"], 4), "val_f1_macro": round(r["f1_macro"], 4),
                    "val_critical_recall_min": round(r["critical_recall_min"], 4),
                    "val_segment_accuracy": round(ev["segment"]["accuracy"], 4),
                    "lr": float(f"{lr:.3g}"), "seconds": round(seconds, 1)})
    print(f"    {phase:12s} epoch {epoch:3d}  loss {loss:.4f}  val acc {r['accuracy']:.3f}  "
          f"macro F1 {r['f1_macro']:.3f}  min critical recall {r['critical_recall_min']:.3f}  ({seconds:.0f} s)",
          flush=True)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def next_version(variant: str) -> str:
    nums = []
    for p in (REPORTS_DIR / "gtm_model").glob("gtm-v*"):
        m = re.match(r"gtm-v(\d+)", p.name)
        if m:
            nums.append(int(m.group(1)))
    try:  # continue the numbering of the model currently in gtm_model/model/
        m = re.search(r"_v(\d+)$", json.loads((GTM_MODEL_DIR / "metadata.json").read_text())["modelName"])
        if m:
            nums.append(int(m.group(1)))
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return f"gtm-v{(max(nums) + 1) if nums else 1}-{variant}"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def write_summary(path: Path, card: dict, comparison: list[dict]) -> None:
    t = card["metrics"]["test"]["recording"]
    lines = [
        f"# GTM model {card['version']}",
        "",
        f"* Network: **{card['architecture']}**",
        f"* Trained: {card['trained_at']} with `scripts/train_gtm_model.py`",
        f"* Input: {card['input']}",
        f"* Training data: {card['dataset']['train_windows']} windows from {card['dataset']['train_recordings']} "
        f"train-split recordings (originals + augmented copies); validation {card['dataset']['validation_recordings']} "
        f"and test {card['dataset']['test_recordings']} original recordings",
        f"* Critical-class weights (folded into the output bias): `{json.dumps(card['class_weights'])}`",
        "",
        "## Test set (recording level, exported model files)",
        "",
        "| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |",
        "|---|---|---|---|---|",
        f"| {t['accuracy']:.3f} | {t['precision_macro']:.3f} | {t['recall_macro']:.3f} | {t['f1_macro']:.3f} | "
        f"{t['critical_recall_min']:.3f} |",
        "",
        "## Class-wise performance (test)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for cls, m in t["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    lines += ["", "## Model comparison", "",
              "| Model | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall (test) | Selected |",
              "|---|---|---|---|---|---|---|"]
    for c in comparison:
        lines.append(f"| {c['label']} | {c['val_accuracy']:.3f} | {c['val_f1_macro']:.3f} | {c['test_accuracy']:.3f} | "
                     f"{c['test_f1_macro']:.3f} | {c['test_critical_recall_min']:.3f} | {'yes' if c['selected'] else ''} |")
    live = card["metrics"].get("live_microphone_path") or {}
    if live:
        lines += ["", "## Live-microphone path (raw audio, no noise reduction or trimming)", "",
                  "| Split | Accuracy | Macro F1 |", "|---|---|---|"]
        for split, m in live.items():
            lines.append(f"| {split} | {m['accuracy']:.3f} | {m['f1_macro']:.3f} |")
    lines += ["", "## Export check", "",
              f"The exported model.json / weights.bin were run with the NumPy implementation in "
              f"`gtm_model/inference.py`: max |probability difference| to the trained network = "
              f"{card['export_check']['max_abs_diff']:.2e} on {card['export_check']['windows']} validation windows."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30, help="max fine-tuning epochs (early stopping on validation)")
    parser.add_argument("--transfer-epochs", type=int, default=50, help="Teachable Machine transfer epochs")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4, help="fine-tuning learning rate")
    parser.add_argument("--no-live-augment", action="store_true",
                        help="train on the upload rendering only (not the live-microphone rendering)")
    parser.add_argument("--no-finetune", action="store_true", help="only Teachable Machine's frozen-base transfer")
    parser.add_argument("--no-critical-weighting", action="store_true")
    parser.add_argument("--no-export", action="store_true", help="do not replace gtm_model/model/")
    parser.add_argument("--quick", action="store_true", help="small, fast run to check the pipeline")
    parser.add_argument("--rebuild-cache", action="store_true", help="recompute the cached GTM windows")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=0, help="PyTorch CPU threads (0 = PyTorch default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from data/features/gtm/train_checkpoint.pt")
    args = parser.parse_args(argv)
    try:
        import torch
    except ImportError:
        print("PyTorch is needed to train the GTM model:  python -m pip install -r requirements-gtm.txt")
        return 1
    if args.quick:
        args.epochs, args.transfer_epochs, args.patience = 2, 3, 2
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    t_start = time.time()

    classes = class_names()
    index = {c: i for i, c in enumerate(classes)}
    settings = load_default_settings()
    meta = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)

    print("GTM inputs (exactly what static/js/gtm.js computes in the browser):", flush=True)
    names = ["train-upload", "validation-upload", "test-upload", "validation-live", "test-live"]
    if not args.no_live_augment:
        names.insert(1, "train-live")
    sets = load_sets(meta, settings, names, args.jobs, args.rebuild_cache)
    tr_ctx, tr_meta = sets["train-upload"]
    va_X, va_meta = sets["validation-upload"]
    te_X, te_meta = sets["test-upload"]
    # Leakage guard: a root recording may only appear in one split.
    roots = {k: set(v[1]["root_audio_id"]) for k, v in sets.items()}
    assert not (roots["train-upload"] & roots["validation-upload"]), "train/validation leakage"
    assert not (roots["train-upload"] & roots["test-upload"]), "train/test leakage"
    assert not (roots["validation-upload"] & roots["test-upload"]), "validation/test leakage"
    y_tr = tr_meta["class_label"].map(index).to_numpy()
    if args.quick:
        keep = np.random.default_rng(args.seed).permutation(len(y_tr))[:3000]
        tr_ctx, tr_meta, y_tr = tr_ctx[keep], tr_meta.iloc[keep].reset_index(drop=True), y_tr[keep]

    print("\nSpeech-commands base model (Teachable Machine's audio network):", flush=True)
    base = ensure_base_model()
    net = build_network(len(classes))
    # Conversion check: our PyTorch copy of the base must reproduce the TensorFlow.js base model.
    check = build_network(len(base["weights"]["dense_2/bias"]))
    load_base_weights(check, base["weights"], head="dense_2")
    sample = va_X[:64]
    diff = float(np.abs(predict(check, sample) - TFJSLayersModel.load(BASE_DIR).predict(sample)).max())
    print(f"  PyTorch copy vs TensorFlow.js layout: max |p difference| = {diff:.2e}", flush=True)
    if diff > 1e-4:
        raise RuntimeError("Base model conversion check failed.")
    load_base_weights(net, base["weights"])

    history: list[dict] = []
    results: dict = {}
    states: dict = {}
    t_split = {}
    # Progress is saved after every epoch, so an interrupted run can continue with --resume.
    ckpt_path = CACHE_DIR / "train_checkpoint.pt"
    run_key = {"data": _fingerprint(meta, settings), "epochs": args.epochs, "transfer_epochs": args.transfer_epochs,
               "lr": args.lr, "patience": args.patience, "seed": args.seed, "live": not args.no_live_augment,
               "finetune": not args.no_finetune, "quick": bool(args.quick)}
    ckpt: dict = {"key": run_key}
    if args.resume and ckpt_path.exists():
        saved = torch.load(ckpt_path, weights_only=False)
        if saved.get("key") == run_key:
            ckpt = saved
            history = ckpt["history"]
        else:
            print("  (checkpoint was made with other settings or data - starting from the beginning)", flush=True)

    if "tm_transfer" in ckpt:
        states["tm_transfer"], t_split["tm_transfer"] = ckpt["tm_transfer"], ckpt["transfer_seconds"]
        print("\n[1] tm-transfer: restored from the checkpoint", flush=True)
    else:
        print(f"\n[1] tm-transfer: frozen base + new softmax layer ({len(y_tr)} windows, "
              f"{args.transfer_epochs} epochs)", flush=True)
        t0 = time.time()
        states["tm_transfer"] = train_transfer(net, tr_ctx, y_tr, va_X, va_meta, classes, args.transfer_epochs,
                                               args.seed, history)
        t_split["tm_transfer"] = time.time() - t0
        ckpt.update(tm_transfer=states["tm_transfer"], transfer_seconds=t_split["tm_transfer"], history=history)
        if not args.quick:
            save_checkpoint(ckpt, ckpt_path)
    net.load_state_dict(states["tm_transfer"])

    if not args.no_finetune:
        pools, labels = [tr_ctx], [y_tr]
        if "train-live" in sets:
            live_ctx, live_meta = sets["train-live"]
            y_live = live_meta["class_label"].map(index).to_numpy()
            if args.quick:
                keep = np.random.default_rng(args.seed + 1).permutation(len(y_live))[:3000]
                live_ctx, y_live = live_ctx[keep], y_live[keep]
            pools.append(live_ctx)
            labels.append(y_live)
        n = sum(len(p) for p in pools)
        print(f"\n[2] tm-finetuned: whole network, {n} windows per epoch "
              f"(upload{' + live-microphone' if len(pools) > 1 else ''} renderings), up to {args.epochs} epochs",
              flush=True)
        t0 = time.time()
        states["tm_finetuned"] = train_finetune(net, pools, labels, va_X, va_meta, classes, args.epochs,
                                                args.patience, args.lr, args.seed, history, ckpt,
                                                None if args.quick else ckpt_path)
        t_split["tm_finetuned"] = time.time() - t0

    labels_txt = {"tm_transfer": "Teachable Machine transfer (frozen speech-commands base + new softmax layer)",
                  "tm_finetuned": "Teachable Machine network fine-tuned end to end"}
    for name, state in states.items():
        net.load_state_dict(state)
        results[name] = {"label": labels_txt[name], "validation": evaluate(predict(net, va_X), va_meta, classes),
                         "test": evaluate(predict(net, te_X), te_meta, classes), "train_seconds": t_split[name]}
    try:  # the model currently in gtm_model/model/, evaluated on the same inputs
        prev = TFJSLayersModel.load(GTM_MODEL_DIR)
        prev_meta = json.loads((GTM_MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
        prev_labels = prev_meta.get("wordLabels") or prev_meta.get("labels")
        order = [prev_labels.index(c) for c in classes]
        results["previous"] = {"label": f"Previous export ({prev_meta.get('modelName', 'gtm')})",
                               "validation": evaluate(prev.predict(va_X)[:, order], va_meta, classes),
                               "test": evaluate(prev.predict(te_X)[:, order], te_meta, classes), "train_seconds": 0.0}
    except Exception as exc:  # pragma: no cover - no previous model
        print(f"  (previous GTM model not evaluated: {exc})")

    best_name = max(states, key=lambda n: rank_key(results[n]["validation"]))
    variant = best_name.replace("_", "-")
    net.load_state_dict(states[best_name])
    print(f"\nSelected: {results[best_name]['label']} (validation macro F1 "
          f"{results[best_name]['validation']['recording']['f1_macro']:.4f})", flush=True)

    weights = np.ones(len(classes))
    if not args.no_critical_weighting:
        weights, info = tune_critical_weights(predict(net, va_X), va_meta, classes)
        if not np.allclose(weights, 1.0):
            with torch.no_grad():
                net.head.bias.add_(torch.from_numpy(np.log(weights).astype(np.float32)))
            changed = {classes[i]: round(float(weights[i]), 2) for i in range(len(classes)) if weights[i] != 1.0}
            results[f"{best_name}_weighted"] = {
                "label": results[best_name]["label"] + " + critical-class weighting",
                "validation": evaluate(predict(net, va_X), va_meta, classes),
                "test": evaluate(predict(net, te_X), te_meta, classes), "train_seconds": t_split[best_name],
                "class_weights": changed, "validation_effect": info}
            best_name = f"{best_name}_weighted"
            print(f"Critical-class weighting {changed}: validation min critical recall "
                  f"{info['before']['critical_recall_min']:.3f} -> {info['after']['critical_recall_min']:.3f}, "
                  f"macro F1 {info['before']['f1_macro']:.3f} -> {info['after']['f1_macro']:.3f}", flush=True)
        else:
            print("Critical-class weighting: no weights improved validation critical recall; kept the model as is.")

    # ---- confidence calibration: one temperature fitted on the validation windows, folded into the output
    # layer (logits / T), so the browser shows calibrated confidences. The predicted class does not change.
    y_va = va_meta["class_label"].map(index).to_numpy()
    y_te = te_meta["class_label"].map(index).to_numpy()
    p_va_raw, p_te_raw = predict(net, va_X), predict(net, te_X)
    temperature = fit_temperature(p_va_raw, y_va)
    with torch.no_grad():
        net.head.weight.div_(temperature)
        net.head.bias.div_(temperature)
    p_va_cal, p_te_cal = predict(net, va_X), predict(net, te_X)
    _, r_raw, r_y = aggregate_by_recording(p_te_raw, te_meta["audio_id"].to_numpy(), y_te)
    _, r_cal, _ = aggregate_by_recording(p_te_cal, te_meta["audio_id"].to_numpy(), y_te)
    calibration = {
        "method": "temperature scaling fitted on the validation windows, folded into NewHeadDense (logits / T)",
        "temperature": round(temperature, 4),
        "validation_windows": {"before": calibration_summary(p_va_raw, y_va), "after": calibration_summary(p_va_cal, y_va)},
        "test_windows": {"before": calibration_summary(p_te_raw, y_te), "after": calibration_summary(p_te_cal, y_te)},
        "test_recordings": {"before": calibration_summary(r_raw, r_y), "after": calibration_summary(r_cal, r_y)},
    }
    print(f"Calibration: temperature {temperature:.3f}; test window ECE {calibration['test_windows']['before']['ece']:.3f}"
          f" -> {calibration['test_windows']['after']['ece']:.3f}", flush=True)

    # ---- export to a temporary folder and verify the files with the NumPy runner ----
    version = next_version(variant)
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = Path(tempfile.mkdtemp(prefix="gtm_export_"))
    export_tfjs(net, classes, base["model_json"], tmp,
                "SonicSentinel scripts/train_gtm_model.py (speech-commands 18w base + NewHeadDense)")
    exported = TFJSLayersModel.load(tmp)
    torch_p = predict(net, va_X)
    va_p, te_p = exported.predict(va_X), exported.predict(te_X)
    export_diff = float(np.abs(va_p - torch_p).max())
    print(f"\nExport check (NumPy runner on the exported files vs PyTorch): max |p difference| = {export_diff:.2e}",
          flush=True)
    if export_diff > 1e-4:
        raise RuntimeError("The exported model does not reproduce the trained network.")
    final = {"validation": evaluate(va_p, va_meta, classes), "test": evaluate(te_p, te_meta, classes)}
    live = {}
    for split in ("validation", "test"):
        X, m = sets[f"{split}-live"]
        live[split] = evaluate(exported.predict(X), m, classes)["recording"]

    # ---- reports ----
    # --quick runs are only a pipeline check: keep their report out of reports/
    report_dir = tmp / "report" if args.quick else REPORTS_DIR / "gtm_model" / version
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = []
    for name, r in results.items():
        v, t = r["validation"]["recording"], r["test"]["recording"]
        comparison.append({
            "model": name, "label": r["label"], "selected": name == best_name,
            "val_accuracy": round(v["accuracy"], 4), "val_f1_macro": round(v["f1_macro"], 4),
            "val_critical_recall_min": round(v["critical_recall_min"], 4),
            "test_accuracy": round(t["accuracy"], 4), "test_precision_macro": round(t["precision_macro"], 4),
            "test_recall_macro": round(t["recall_macro"], 4), "test_f1_macro": round(t["f1_macro"], 4),
            "test_critical_recall_min": round(t["critical_recall_min"], 4),
            "test_segment_accuracy": round(r["test"]["segment"]["accuracy"], 4),
            "train_seconds": round(r["train_seconds"], 1),
        })
        plot_confusion_matrix(t["confusion_matrix"], classes, f"GTM - {r['label']} - test set (recording level)",
                              report_dir / f"confusion_matrix_{name}.png")
    pd.DataFrame(comparison).to_csv(report_dir / "model_comparison.csv", index=False)
    pd.DataFrame(history).to_csv(report_dir / "training_history.csv", index=False)

    ids, agg, truth = aggregate_by_recording(te_p, te_meta["audio_id"].to_numpy(),
                                             te_meta["class_label"].map(index).to_numpy())
    samples = []
    for c in range(len(classes)):
        for i in np.flatnonzero(truth == c)[:3]:
            top = np.argsort(agg[i])[::-1][:3]
            samples.append({"audio_id": ids[i], "actual": classes[truth[i]], "predicted": classes[top[0]],
                            "confidence": round(float(agg[i][top[0]]), 4),
                            "top3": "; ".join(f"{classes[j]} {agg[i][j]:.2f}" for j in top),
                            "correct": bool(top[0] == truth[i])})
    pd.DataFrame(samples).to_csv(report_dir / "sample_predictions.csv", index=False)

    card = _jsonable({
        "version": version,
        "variant": best_name,
        "trained_at": trained_at,
        "architecture": "Teachable Machine audio network: speech-commands 18w BROWSER_FFT base "
                        "(4 x Conv2D/MaxPooling2D, Dense 2000) + NewHeadDense softmax",
        "base_model": BASE_URL,
        "input": "43 x 232 spectrogram of the loudest ~1 s of each 2 s segment (44.1 kHz, FFT 2048, hop 1024, "
                 "0-5 kHz), z-score normalised - computed as in static/js/gtm.js",
        "classes": classes,
        "class_weights": {classes[i]: round(float(weights[i]), 2) for i in range(len(classes)) if weights[i] != 1.0},
        "training": {"transfer_epochs": args.transfer_epochs, "finetune_max_epochs": None if args.no_finetune else args.epochs,
                     "finetune_lr": args.lr, "patience": args.patience, "seed": args.seed,
                     "augmentation": [] if args.no_finetune else
                     ["time shift +-10 frames", "SpecAugment (1 frequency + 1 time mask)", "label smoothing 0.05"]
                     + ([] if args.no_live_augment else ["live-microphone rendering of the training recordings"]),
                     "quick": bool(args.quick)},
        "dataset": {"train_windows": int(len(y_tr)), "train_recordings": int(tr_meta["audio_id"].nunique()),
                    "validation_recordings": int(va_meta["audio_id"].nunique()),
                    "test_recordings": int(te_meta["audio_id"].nunique())},
        "metrics": {"validation": final["validation"], "test": final["test"], "live_microphone_path": live},
        "calibration": calibration,
        "export_check": {"max_abs_diff": export_diff, "windows": int(len(va_X))},
        "comparison": comparison,
        "report_dir": str(report_dir if args.quick else report_dir.relative_to(REPORTS_DIR.parent)).replace("\\", "/"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
    })
    (report_dir / "metrics.json").write_text(json.dumps(_jsonable({"results": results, "card": card}), indent=2),
                                             encoding="utf-8")
    write_summary(report_dir / "summary.md", card, comparison)

    t = final["test"]["recording"]
    metadata = {
        "tfjsSpeechCommandsVersion": "0.4.0",
        "modelName": f"SonicSentinel_GTM_{version.split('-')[1]}",
        "timeStamp": trained_at,
        "wordLabels": classes,
        "labels": classes,
        "frameSize": GTM_BINS,
        "version": version,
        "architecture": card["architecture"],
        "accuracy": round(t["accuracy"], 4),
        "metrics": {"test_recording_accuracy": round(t["accuracy"], 4), "test_f1_macro": round(t["f1_macro"], 4),
                    "validation_recording_accuracy": round(final["validation"]["recording"]["accuracy"], 4),
                    "validation_f1_macro": round(final["validation"]["recording"]["f1_macro"], 4)},
        "note": "Trained with scripts/train_gtm_model.py on the project dataset (same splits, pre-processing and "
                "evaluation as the Python model). Accuracy = recording-level accuracy on the unseen test split.",
        "report_dir": card["report_dir"],
    }
    (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.no_export or args.quick:
        print(f"\nNot exported ({'--quick' if args.quick else '--no-export'}); files left in {tmp}")
    else:
        GTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for f in ("model.json", "weights.bin", "metadata.json"):
            shutil.copyfile(tmp / f, GTM_MODEL_DIR / f)
        for extra in GTM_MODEL_DIR.glob("*.bin"):  # an older export may have used differently named shards
            if extra.name != "weights.bin":
                extra.unlink()
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\nExported {version} to {GTM_MODEL_DIR}")
    print(f"Test (recording level): accuracy {t['accuracy']:.3f}, macro F1 {t['f1_macro']:.3f}, "
          f"min critical recall {t['critical_recall_min']:.3f}")
    print(f"Live-microphone path: validation accuracy {live['validation']['accuracy']:.3f}, "
          f"test accuracy {live['test']['accuracy']:.3f}")
    print(f"Reports in {report_dir}  ({(time.time() - t_start) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
