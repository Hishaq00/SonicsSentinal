"""Evaluation helpers shared by training, the comparison report and tests."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

CRITICAL_CLASSES = ["Gunshot", "Glass Breaking", "Panic Scream", "Aggression", "Person Asking for Help", "Explosion"]


def aggregate_by_recording(proba: np.ndarray, groups: np.ndarray, y_true: np.ndarray):
    """Mean segment probability per recording -> (recording ids, proba, true labels)."""
    ids, inverse = np.unique(groups, return_inverse=True)
    agg = np.zeros((len(ids), proba.shape[1]))
    np.add.at(agg, inverse, proba)
    counts = np.bincount(inverse).astype(float)
    agg /= counts[:, None]
    truth = np.empty(len(ids), dtype=y_true.dtype)
    truth[inverse] = y_true
    return ids, agg, truth


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        classes[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i in labels
    }
    critical = [per_class[c]["recall"] for c in CRITICAL_CLASSES if c in per_class and per_class[c]["support"] > 0]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "critical_recall_min": float(min(critical)) if critical else 0.0,
        "critical_recall_mean": float(np.mean(critical)) if critical else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def temper(proba: np.ndarray, temperature: float | None) -> np.ndarray:
    """Temperature scaling: softmax(log p / T). T > 1 softens over-confident probabilities, T < 1 sharpens them;
    the order of the classes, and therefore every prediction, stays the same."""
    p = np.asarray(proba, dtype=np.float64)
    if not temperature or abs(float(temperature) - 1.0) < 1e-9:
        return p
    z = np.log(np.clip(p, 1e-12, 1.0)) / float(temperature)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y_true: np.ndarray) -> float:
    """The temperature with the lowest log loss on held-out predictions (the validation split)."""
    from scipy.optimize import minimize_scalar

    y = np.asarray(y_true, dtype=int)
    rows = np.arange(len(y))

    def nll(log_t: float) -> float:
        return float(-np.mean(np.log(np.clip(temper(proba, float(np.exp(log_t)))[rows, y], 1e-12, 1.0))))

    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded").x))


def calibration_summary(proba: np.ndarray, y_true: np.ndarray, bins: int = 10) -> dict:
    """How well the confidence matches the accuracy: expected calibration error (ECE), mean confidence,
    accuracy and log loss."""
    p = np.asarray(proba, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return {"ece": round(float(ece), 4), "mean_confidence": round(float(conf.mean()), 4),
            "accuracy": round(float(correct.mean()), 4),
            "log_loss": round(float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))), 4)}


SHORT_NAMES = {
    "Person Asking for Help": "Help", "Background Noise": "Background", "Alarm or Siren": "Alarm/Siren",
    "Machinery Fault": "Machinery", "Drilling or Grinder Sound": "Drill/Grinder", "Vehicle Backfire": "Backfire",
    "Graffiti Spray": "Graffiti", "Door Impact": "Door", "Drone Sound": "Drone",
    "Human Speech": "Speech", "Human Body Sound": "Body", "Laughter or Applause": "Laugh/Clap",
    "Keyboard or Mouse": "Keyboard", "Household Appliance": "Appliance", "Rain or Water": "Rain/Water",
    "Fire Crackling": "Fire", "Church Bell": "Bell", "Train or Aircraft": "Train/Air",
    "Engine Running": "Engine", "Crying Baby": "Baby", "Thunderstorm": "Thunder",
}


def plot_confusion_matrix(cm: list[list[int]], classes: list[str], title: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    k = len(classes)
    fig, ax = plt.subplots(figsize=(max(8.6, 0.55 * k + 3), max(7.4, 0.5 * k + 2.4)), dpi=110)
    ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
    short = [SHORT_NAMES.get(c, c) for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10, loc="left")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "#1d1c1a")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
