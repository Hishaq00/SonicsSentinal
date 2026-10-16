"""Log-mel convolutional network (CNN) member of the Python classifier.

The CNN looks at the 64-band log-mel spectrogram of each 2-second segment
(22,050 Hz, n_fft 1024, hop 512 -> 64 x 87 values). It is trained with PyTorch
(``python -m python_models.train_cnn``, needs ``requirements-gtm.txt``); the web
application runs the trained network with the NumPy forward pass below, so
PyTorch is not needed at run time.

Network: 4 blocks of [3x3 conv - batch norm - ReLU] x 2 + 2x2 max-pool
(16, 32, 64, 128 channels), then global mean + max pooling over frequency and
time, dropout and one dense layer with 10 outputs. For inference every batch-norm
layer is folded into the convolution before it (w' = w * gamma / sqrt(var + eps),
b' = beta - mean * gamma / sqrt(var + eps)), so a block is conv -> ReLU.

This model is trained on our own 22 kHz log-mel input and never sees the GTM
model or its output, so the Python and GTM predictions stay independent.
"""
from __future__ import annotations

import numpy as np

LOGMEL_CONFIG = {"sr": 22050, "n_fft": 1024, "hop_length": 512, "n_mels": 64, "fmin": 20.0, "min_frames": 32}


def logmel(y: np.ndarray, cfg: dict = LOGMEL_CONFIG) -> np.ndarray:
    """(n_mels, frames) log-mel power in dB of one mono segment at ``cfg['sr']``."""
    import librosa  # imported lazily: librosa takes several seconds to import

    y = np.asarray(y, dtype=np.float32)
    if y.size < cfg["n_fft"]:
        y = np.pad(y, (0, cfg["n_fft"] - y.size))
    mel = librosa.feature.melspectrogram(y=y, sr=cfg["sr"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
                                         n_mels=cfg["n_mels"], fmin=cfg["fmin"], fmax=cfg["sr"] / 2, power=2.0)
    db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None).astype(np.float32)
    db = np.clip(db, -100.0, 20.0)
    if db.shape[1] < cfg["min_frames"]:  # very short live windows: four 2x2 poolings need >= 16 frames
        db = np.pad(db, ((0, 0), (0, cfg["min_frames"] - db.shape[1])), constant_values=-100.0)
    return db


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _conv3x3_relu(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 convolution, padding 1, stride 1, then ReLU. x: (N, C, H, W), w: (O, C, 3, 3)."""
    n, c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))  # (N, C, H, W, 3, 3)
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * wd, c * 9)
    out = cols @ w.reshape(w.shape[0], -1).T + b
    return np.maximum(out.reshape(n, h, wd, -1).transpose(0, 3, 1, 2), 0.0)


def _maxpool2(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    x = x[:, :, : h // 2 * 2, : w // 2 * 2]
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


class NumpyCNN:
    """Inference with the exported (batch-norm folded) weights."""

    def __init__(self, export: dict):
        self.config = dict(export["config"])
        self.mean = float(export["norm_mean"])
        self.std = float(export["norm_std"])
        self.convs = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in export["convs"]]
        self.fc_w = np.asarray(export["fc_w"], np.float32)
        self.fc_b = np.asarray(export["fc_b"], np.float32)
        self.pool_after = int(export.get("convs_per_block", 2))

    def logits(self, mels: np.ndarray) -> np.ndarray:
        """mels: (N, n_mels, frames) in dB -> (N, n_classes) logits."""
        h = ((np.asarray(mels, np.float32) - self.mean) / self.std)[:, None]
        for i, (w, b) in enumerate(self.convs, start=1):
            h = _conv3x3_relu(h, w, b)
            if i % self.pool_after == 0:
                h = _maxpool2(h)
        pooled = np.concatenate([h.mean(axis=(2, 3)), h.max(axis=(2, 3))], axis=1)
        return pooled @ self.fc_w.T + self.fc_b

    def predict_proba_mels(self, mels: list[np.ndarray], batch: int = 8) -> np.ndarray:
        """Class probabilities for log-mel inputs of possibly different lengths."""
        out = []
        for i in range(0, len(mels), batch):
            chunk = mels[i: i + batch]
            frames = max(m.shape[1] for m in chunk)  # equal lengths inside a batch (segments are all 2 s)
            arr = np.stack([np.pad(m, ((0, 0), (0, frames - m.shape[1])), mode="edge") for m in chunk])
            out.append(softmax(self.logits(arr)))
        return np.vstack(out)

    def predict_proba_segments(self, segments: list[np.ndarray]) -> np.ndarray:
        return self.predict_proba_mels([logmel(s, self.config) for s in segments])


def fuse(p_features: np.ndarray, p_cnn: np.ndarray, cnn_weight: float, critical_weights=None) -> np.ndarray:
    """Weighted mean of the two members' probabilities, then the critical-class decision weights."""
    p = (1.0 - cnn_weight) * np.asarray(p_features, float) + cnn_weight * np.asarray(p_cnn, float)
    if critical_weights is not None:
        p = p * np.asarray(critical_weights, float)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
