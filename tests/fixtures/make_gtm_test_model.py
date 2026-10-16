"""Create a GTM-*format* TensorFlow.js model for automated integration tests.

THIS IS NOT A TEACHABLE MACHINE MODEL. It has random weights, so its
predictions are meaningless. It only has the same file format, input shape
(43 x 232 x 1) and label list as an exported Teachable Machine audio model,
which lets the tests exercise the browser GTM pipeline (loading, spectrogram,
prediction, label mapping, posting results) without a real GTM export.

    python tests/fixtures/make_gtm_test_model.py
Writes tests/fixtures/gtm_test_model/{model.json, weights.bin, metadata.json}.
Never copy these files into gtm_model/model/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import class_names  # noqa: E402

OUT = Path(__file__).resolve().parent / "gtm_test_model"
FRAMES, BINS = 43, 232


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = class_names()
    n_in, n_out = FRAMES * BINS, len(labels)
    rng = np.random.default_rng(2026)
    kernel = (rng.normal(0, 0.02, (n_in, n_out))).astype("<f4")
    bias = np.zeros(n_out, dtype="<f4")
    (OUT / "weights.bin").write_bytes(kernel.tobytes() + bias.tobytes())
    topology = {
        "class_name": "Sequential",
        "config": {"name": "fixture_sequential", "layers": [
            {"class_name": "Flatten", "config": {"name": "fixture_flatten", "trainable": True,
                                                 "batch_input_shape": [None, FRAMES, BINS, 1], "dtype": "float32",
                                                 "data_format": "channels_last"}},
            {"class_name": "Dense", "config": {"name": "fixture_dense", "trainable": True, "units": n_out,
                                               "activation": "softmax", "use_bias": True,
                                               "kernel_initializer": {"class_name": "Zeros", "config": {}},
                                               "bias_initializer": {"class_name": "Zeros", "config": {}},
                                               "kernel_regularizer": None, "bias_regularizer": None,
                                               "activity_regularizer": None, "kernel_constraint": None,
                                               "bias_constraint": None}},
        ]},
        "keras_version": "tfjs-layers 4.22.0",
        "backend": "tensor_flow.js",
    }
    model = {
        "format": "layers-model",
        "generatedBy": "SonicSentinel test fixture (random weights, NOT a Teachable Machine model)",
        "convertedBy": None,
        "modelTopology": topology,
        "weightsManifest": [{"paths": ["weights.bin"], "weights": [
            {"name": "fixture_dense/kernel", "shape": [n_in, n_out], "dtype": "float32"},
            {"name": "fixture_dense/bias", "shape": [n_out], "dtype": "float32"},
        ]}],
    }
    (OUT / "model.json").write_text(json.dumps(model), encoding="utf-8")
    meta = {
        "tfjsSpeechCommandsVersion": "0.4.0",
        "modelName": "integration-test-fixture-NOT-GTM",
        "timeStamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wordLabels": labels,
        "note": "Random weights. Format-compatible test fixture only; not a Teachable Machine model.",
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Test fixture written to {OUT}")


if __name__ == "__main__":
    main()
