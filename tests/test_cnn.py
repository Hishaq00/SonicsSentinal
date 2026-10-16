"""Log-mel CNN member: input shape, NumPy inference == PyTorch, fusion."""
from __future__ import annotations

import numpy as np
import pytest

from python_models.cnn import LOGMEL_CONFIG, NumpyCNN, fuse, logmel

from .conftest import SR, noise, tone


def test_logmel_shape_for_a_two_second_segment():
    m = logmel(tone(2.0, sr=SR))
    assert m.shape == (LOGMEL_CONFIG["n_mels"], 87)
    assert np.isfinite(m).all() and m.max() <= 20.0 and m.min() >= -100.0


def test_short_live_window_is_padded_for_the_poolings():
    m = logmel(noise(0.3, sr=SR))
    assert m.shape[1] >= LOGMEL_CONFIG["min_frames"]


def test_fusion_weights_and_normalisation():
    pf = np.array([[0.7, 0.2, 0.1]])
    pc = np.array([[0.1, 0.8, 0.1]])
    assert np.allclose(fuse(pf, pc, 0.0), pf)
    assert np.allclose(fuse(pf, pc, 1.0), pc)
    mixed = fuse(pf, pc, 0.5, [1.0, 1.0, 3.0])
    assert np.isclose(mixed.sum(), 1.0) and mixed[0, 2] > 0.1


def test_numpy_inference_matches_pytorch():
    torch = pytest.importorskip("torch")
    from python_models.train_cnn import build_torch_model, export_numpy

    torch.manual_seed(0)
    model = build_torch_model(10)
    with torch.no_grad():  # non-trivial batch-norm statistics, so the folding is really tested
        for layer in model.features:
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.uniform_(-0.5, 0.5)
                layer.running_var.uniform_(0.5, 2.0)
                layer.weight.uniform_(0.5, 1.5)
                layer.bias.uniform_(-0.2, 0.2)
    model.eval()
    mels = np.stack([logmel(tone(2.0, 440 * (i + 1), sr=SR)) for i in range(3)] + [logmel(noise(2.0, sr=SR))])
    mean, std = float(mels.mean()), float(mels.std())
    with torch.no_grad():
        ref = torch.softmax(model(torch.from_numpy(((mels - mean) / std)[:, None].astype(np.float32))), 1).numpy()
    ours = NumpyCNN(export_numpy(model, mean, std)).predict_proba_mels(list(mels))
    assert np.abs(ours - ref).max() < 1e-4
