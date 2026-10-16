"""Run an exported GTM model (TensorFlow.js layers format) in Python.

The web application runs the GTM model in the browser with TensorFlow.js.
This module is a small NumPy implementation of the layers Teachable Machine
audio models use (Conv2D, MaxPooling2D, Flatten, Dropout, Dense), so Python
code can run the *same exported files*: the training script uses it to check
the export, and the comparison report and tests use it to get real GTM
predictions without a browser.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config import GTM_MODEL_DIR
from feature_extraction.gtm_spectrogram import GTM_BINS, GTM_FRAMES


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


_ACTIVATIONS = {
    None: lambda x: x,
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(x, 0.0),
    "softmax": _softmax,
}


def _flatten_layers(topology: dict) -> list[dict]:
    """Layer configs of a Sequential model (Keras 2 list or dict form); nested Sequentials are inlined."""
    if "model_config" in topology:
        topology = topology["model_config"]
    cfg = topology.get("config")
    layers = cfg if isinstance(cfg, list) else (cfg or {}).get("layers", [])
    out: list[dict] = []
    for layer in layers:
        if layer["class_name"] in ("Sequential", "Model", "Functional"):
            out += _flatten_layers(layer)
        elif layer["class_name"] != "InputLayer":
            out.append(layer)
    return out


def read_weights(folder: Path, manifest: list[dict]) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    for group in manifest:
        data = b"".join((folder / p).read_bytes() for p in group["paths"])
        offset = 0
        for spec in group["weights"]:
            if spec.get("dtype", "float32") != "float32" or spec.get("quantization"):
                raise ValueError(f"Unsupported weight encoding for {spec['name']} (only float32 is supported).")
            n = int(np.prod(spec["shape"])) if spec["shape"] else 1
            arr = np.frombuffer(data, dtype="<f4", count=n, offset=offset).reshape(spec["shape"])
            weights[spec["name"]] = arr.astype(np.float32)
            offset += 4 * n
        if offset != len(data):
            raise ValueError(f"Weight files hold {len(data)} bytes but the manifest describes {offset}.")
    return weights


class TFJSLayersModel:
    """Minimal NumPy inference engine for Sequential TensorFlow.js layers models."""

    SUPPORTED = ("Conv2D", "MaxPooling2D", "Flatten", "Dropout", "Dense", "Activation")

    def __init__(self, layers: list[dict], weights: dict[str, np.ndarray]):
        self.layers = layers
        self.input_shape = tuple(layers[0]["config"].get("batch_input_shape") or [None, GTM_FRAMES, GTM_BINS, 1])[1:]
        self.params: list[dict] = []
        for layer in layers:
            kind, cfg = layer["class_name"], layer["config"]
            if kind not in self.SUPPORTED:
                raise ValueError(f"Layer type {kind} is not supported by the Python GTM runner.")
            if kind in ("Conv2D", "MaxPooling2D") and cfg.get("padding", "valid") != "valid":
                raise ValueError(f"{cfg['name']}: only 'valid' padding is supported.")
            if cfg.get("data_format", "channels_last") != "channels_last":
                raise ValueError(f"{cfg['name']}: only channels_last is supported.")
            p = {}
            if kind in ("Conv2D", "Dense"):
                p["kernel"] = self._weight(weights, cfg["name"], "kernel")
                p["bias"] = self._weight(weights, cfg["name"], "bias") if cfg.get("use_bias", True) else None
            self.params.append(p)

    @staticmethod
    def _weight(weights: dict[str, np.ndarray], layer_name: str, kind: str) -> np.ndarray:
        suffix = f"{layer_name}/{kind}"
        for name, arr in weights.items():
            if name == suffix or name.endswith("/" + suffix):
                return arr
        raise KeyError(f"Weight {suffix} not found in the weights manifest.")

    @classmethod
    def load(cls, folder: str | Path) -> "TFJSLayersModel":
        folder = Path(folder)
        model_json = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        return cls(_flatten_layers(model_json["modelTopology"]), read_weights(folder, model_json["weightsManifest"]))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        for layer, p in zip(self.layers, self.params):
            kind, cfg = layer["class_name"], layer["config"]
            if kind == "Conv2D":
                kh, kw, cin, cout = p["kernel"].shape
                sh, sw = cfg.get("strides", [1, 1])
                win = sliding_window_view(x, (kh, kw), axis=(1, 2))[:, ::sh, ::sw]  # (N, H, W, C, kh, kw)
                n, h, w = win.shape[:3]
                cols = win.transpose(0, 1, 2, 4, 5, 3).reshape(n * h * w, kh * kw * cin)
                x = (cols @ p["kernel"].reshape(kh * kw * cin, cout)).reshape(n, h, w, cout)
                if p["bias"] is not None:
                    x = x + p["bias"]
                x = _ACTIVATIONS[cfg.get("activation")](x)
            elif kind == "MaxPooling2D":
                ph, pw = cfg["pool_size"]
                sh, sw = cfg.get("strides") or cfg["pool_size"]
                x = sliding_window_view(x, (ph, pw), axis=(1, 2))[:, ::sh, ::sw].max(axis=(-2, -1))
            elif kind == "Flatten":
                x = x.reshape(x.shape[0], -1)
            elif kind == "Dense":
                x = x @ p["kernel"]
                if p["bias"] is not None:
                    x = x + p["bias"]
                x = _ACTIVATIONS[cfg.get("activation")](x)
            elif kind == "Activation":
                x = _ACTIVATIONS[cfg.get("activation")](x)
            # Dropout: identity at inference
        return x

    def predict(self, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = x.reshape((-1, *self.input_shape))
        return np.vstack([self._forward(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


class GTMClassifier:
    """The exported GTM model in ``gtm_model/model/`` with its labels, like the browser loads it."""

    def __init__(self, folder: str | Path = GTM_MODEL_DIR):
        self.folder = Path(folder)
        self.metadata = json.loads((self.folder / "metadata.json").read_text(encoding="utf-8"))
        self.labels: list[str] = list(self.metadata.get("wordLabels") or self.metadata.get("words")
                                      or self.metadata.get("labels") or [])
        self.model = TFJSLayersModel.load(self.folder)
        if tuple(self.model.input_shape[:2]) != (GTM_FRAMES, GTM_BINS):
            raise ValueError(f"Unexpected GTM input shape {self.model.input_shape}.")

    @property
    def version(self) -> str:
        """Same version string the browser reports (static/js/gtm.js)."""
        return f"{self.metadata.get('modelName') or 'tm-audio-model'}@{self.metadata.get('timeStamp') or 'unknown'}"

    def predict_inputs(self, inputs: list[np.ndarray] | np.ndarray) -> list[dict[str, float]]:
        """Class -> probability for each (43, 232) normalised input."""
        if len(inputs) == 0:
            return []
        proba = self.model.predict(np.stack(list(inputs)))
        return [{label: float(row[i]) for i, label in enumerate(self.labels)} for row in proba]
