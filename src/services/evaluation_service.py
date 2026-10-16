"""Model Prediction and Confidence Comparison runs (SRS deliverable 6).

An administrator starts a run on the *unseen test split* (at least 10
recordings per class, 100+ in total). Each recording goes through exactly the
same pipeline as a user upload: the server runs the Python model and the
browser runs the GTM model on the same segments. The report then compares the
two models recording by recording and summarises accuracy, macro F1,
critical-class recall, agreement and confidence differences.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from config import DATASET_DIR, METADATA_CSV, class_names, unknown_label
from python_models.evaluation import CRITICAL_CLASSES as CRITICAL  # one list for training and evaluation runs

from ..extensions import db
from ..models import AudioRecord, EvaluationRun, utcnow
from ..utils import next_code
from . import audit, pipeline

SIMILAR = {
    frozenset({"Gunshot", "Glass Breaking"}): "both are short, broadband impulsive sounds",
    frozenset({"Gunshot", "Machinery Fault"}): "knocking machinery also produces sharp repeated impulses",
    frozenset({"Panic Scream", "Aggression"}): "both are loud, strained human voices",
    frozenset({"Person Asking for Help", "Aggression"}): "both contain shouted speech",
    frozenset({"Person Asking for Help", "Panic Scream"}): "both are distressed human vocalisations",
    frozenset({"Alarm or Siren", "Vehicle Horn"}): "both are loud tonal signals with strong harmonics",
    frozenset({"Machinery Fault", "Background Noise"}): "continuous machine noise resembles broadband ambience",
    frozenset({"Animal Sound", "Panic Scream"}): "both are high-pitched vocalisations",
    frozenset({"Animal Sound", "Person Asking for Help"}): "short voiced calls share pitch patterns",
    frozenset({"Glass Breaking", "Machinery Fault"}): "metallic ringing partials appear in both",
    frozenset({"Alarm or Siren", "Machinery Fault"}): "a squealing belt is a sustained high tone like an alarm",
    frozenset({"Gunshot", "Fireworks"}): "both are sharp bangs; fireworks add whistles and crackling tails",
    frozenset({"Gunshot", "Vehicle Backfire"}): "a backfire is a single loud pop like a shot, only duller",
    frozenset({"Gunshot", "Explosion"}): "both are blasts; an explosion has a deeper boom and a longer rumble",
    frozenset({"Explosion", "Fireworks"}): "firework bursts are small explosions",
    frozenset({"Gunshot", "Door Impact"}): "a slammed or kicked door is a sharp impact too",
    frozenset({"Door Impact", "Background Noise"}): "everyday knocks and door closings also occur in ambient sound",
    frozenset({"Drilling or Grinder Sound", "Machinery Fault"}): "power tools and faulty machines are both motor noise",
    frozenset({"Drone Sound", "Drilling or Grinder Sound"}): "both are a motor whine with strong harmonics",
    frozenset({"Drone Sound", "Machinery Fault"}): "rotor buzz resembles a rotating machine",
    frozenset({"Graffiti Spray", "Background Noise"}): "spray hiss resembles broadband noise such as wind or rain",
    frozenset({"Vehicle Backfire", "Vehicle Horn"}): "both come from vehicles and often include engine sound",
    # everyday sound categories (28 Sep 2026)
    frozenset({"Human Speech", "Aggression"}): "the same voice, only calm instead of shouted",
    frozenset({"Human Speech", "Person Asking for Help"}): "both are speech; only the words differ",
    frozenset({"Human Speech", "Background Noise"}): "distant or murmured talking blends into room ambience",
    frozenset({"Human Speech", "Human Body Sound"}): "both come from a person close to the microphone",
    frozenset({"Crying Baby", "Panic Scream"}): "both are long, high-pitched distress cries",
    frozenset({"Crying Baby", "Animal Sound"}): "a wailing infant and some animal calls share pitch contours",
    frozenset({"Human Body Sound", "Background Noise"}): "breathing and coughing are quiet, close-range noise",
    frozenset({"Laughter or Applause", "Human Speech"}): "laughter is voiced like speech",
    frozenset({"Laughter or Applause", "Fireworks"}): "applause is a dense crackle like a firecracker string",
    frozenset({"Footsteps", "Door Impact"}): "both are soft repeated impacts",
    frozenset({"Footsteps", "Background Noise"}): "footsteps are a common part of ordinary ambience",
    frozenset({"Keyboard or Mouse", "Background Noise"}): "small office clicks carry no event of interest",
    frozenset({"Keyboard or Mouse", "Graffiti Spray"}): "both are quiet, dry, high-frequency clatter",
    frozenset({"Household Appliance", "Machinery Fault"}): "an appliance motor and a faulty machine are both motor hum",
    frozenset({"Household Appliance", "Background Noise"}): "appliance hum is part of indoor ambience",
    frozenset({"Rain or Water", "Graffiti Spray"}): "running water and spray hiss are both broadband noise",
    frozenset({"Rain or Water", "Background Noise"}): "rain is a common ambient bed",
    frozenset({"Wind", "Rain or Water"}): "both are broadband weather noise",
    frozenset({"Wind", "Background Noise"}): "wind is a common ambient bed",
    frozenset({"Thunderstorm", "Explosion"}): "both are deep booms with a long rumbling tail",
    frozenset({"Thunderstorm", "Fireworks"}): "distant thunder and firework shells both boom and roll",
    frozenset({"Fire Crackling", "Rain or Water"}): "crackling flames and falling water are both irregular hiss",
    frozenset({"Fire Crackling", "Fireworks"}): "both are strings of small crackles",
    frozenset({"Church Bell", "Alarm or Siren"}): "both are loud tonal signals that repeat",
    frozenset({"Church Bell", "Glass Breaking"}): "both ring with metallic or glassy partials",
    frozenset({"Helicopter", "Drone Sound"}): "both are rotor craft; a helicopter beats much lower and louder",
    frozenset({"Helicopter", "Machinery Fault"}): "rotor beating resembles a knocking machine",
    frozenset({"Train or Aircraft", "Background Noise"}): "a passing train or aeroplane is a rising ambient roar",
    frozenset({"Train or Aircraft", "Drone Sound"}): "aircraft and drones are both sustained propulsion noise",
    frozenset({"Engine Running", "Machinery Fault"}): "a healthy and a faulty engine differ only in the fault",
    frozenset({"Engine Running", "Vehicle Backfire"}): "a backfire happens while the engine is running",
    frozenset({"Engine Running", "Background Noise"}): "idling traffic is a standard ambient hum",
}


def test_pool() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    return meta[(meta["split"] == "test") & (meta["status"] == "original")]


def create_run(user, per_class: int = 10, seed: int = 11) -> EvaluationRun:
    per_class = max(10, min(int(per_class), 200))
    pool = test_pool()
    if pool.empty:
        raise ValueError("No test split found. Build the dataset first (python -m scripts.build_dataset).")
    rng = np.random.default_rng(seed)
    queue = []
    for cls in class_names():
        ids = sorted(pool[pool["class_label"] == cls]["audio_id"].tolist())
        rng.shuffle(ids)
        queue.extend(ids[:per_class])
    from python_models.inference import get_classifier

    run = EvaluationRun(run_code=next_code("EVL"), created_by=user.id, per_class=per_class, total=len(queue),
                        python_version=get_classifier().version, summary={"queue": queue, "failed": []})
    db.session.add(run)
    db.session.flush()
    audit.log("EVALUATION", "evaluation", run.run_code, {"per_class": per_class, "total": len(queue)})
    db.session.commit()
    return run


def run_records(run: EvaluationRun) -> list[AudioRecord]:
    return AudioRecord.query.filter_by(evaluation_run_id=run.id).order_by(AudioRecord.id).all()


def process_next(run: EvaluationRun, user) -> dict:
    """Analyse the next queued test recording; returns the GTM job for the browser."""
    summary = dict(run.summary or {})
    done = {r.dataset_audio_id for r in run_records(run)} | {f["audio_id"] for f in summary.get("failed", [])}
    queue = [a for a in summary.get("queue", []) if a not in done]
    if not queue:
        return {"done": True}
    pool = test_pool().set_index("audio_id")
    aid = queue[0]
    row = pool.loc[aid].to_dict()
    row["audio_id"] = aid
    try:
        rec = pipeline.analyze_file(DATASET_DIR / row["relative_path"], row["filename"], user, source="evaluation",
                                    evaluation_run=run, dataset_row=row, keep_source=True)
    except (pipeline.UploadRejected, pipeline.ProcessingError) as exc:
        summary.setdefault("failed", []).append({"audio_id": aid, "error": str(exc)})
        run.summary = summary
        db.session.commit()
        return {"done": False, "failed": aid, "error": str(exc)}
    run.completed = len(run_records(run))
    db.session.commit()
    return {"done": False, "audio_id": rec.audio_id, "dataset_audio_id": aid, "actual": row["class_label"],
            "gtm_job": pipeline.gtm_job(rec) if rec.gtm_status == "pending" else None,
            "completed": run.completed, "total": run.total}


def explain(rec: AudioRecord) -> str:
    """Plain-language explanation of a model disagreement."""
    if rec.class_match is None or rec.class_match:
        return ""
    py, gtm, actual = rec.python_top, rec.gtm_top, rec.actual_class
    parts = [f"Python chose {py} ({(rec.python_confidence or 0):.0%}) while GTM chose {gtm} ({(rec.gtm_confidence or 0):.0%})."]
    if actual:
        if py == actual:
            parts.append("The Python model was correct.")
        elif gtm == actual:
            parts.append("The GTM model was correct.")
        else:
            parts.append(f"Both were wrong (actual: {actual}).")
    reason = SIMILAR.get(frozenset({py, gtm}))
    if reason:
        parts.append(f"The classes are acoustically similar: {reason}.")
    if rec.quality_label in ("Poor", "Unusable"):
        issues = ", ".join(i["label"] for i in (rec.quality or {}).get("issues", [])[:3])
        parts.append(f"Audio quality was {rec.quality_label}" + (f" ({issues})." if issues else "."))
    if rec.overlapping:
        parts.append(f"Several classes scored highly (possible overlap: {', '.join(rec.overlap_classes or [])}).")
    d = rec.decision or {}
    small = [n for n, m in (("Python", (d.get("python") or {}).get("margin")), ("GTM", (d.get("gtm") or {}).get("margin")))
             if m is not None and m < 0.15]
    if small:
        parts.append(f"{' and '.join(small)} had a small top-two margin (uncertain between two classes).")
    parts.append("GTM sees a 1-second, 0-5 kHz spectrogram; the Python model uses 2-second segments up to 11 kHz, "
                 "so each model relies on different cues.")
    return " ".join(parts)


def _f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    from sklearn.metrics import f1_score

    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _cm(y_true, y_pred, labels):
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=labels).tolist() if y_true else []


def compute_summary(run: EvaluationRun) -> dict:
    recs = [r for r in run_records(run) if r.final_category and r.gtm_status != "pending"]
    labels = class_names()
    actual = [r.actual_class for r in recs]
    py = [r.python_top for r in recs]
    with_gtm = [r for r in recs if r.gtm_top]
    final = [r.final_category for r in recs]

    def acc(pred, truth):
        return float(np.mean([p == t for p, t in zip(pred, truth)])) if truth else None

    def crit_recall(pred, truth):
        out = {}
        for c in CRITICAL:
            idx = [i for i, t in enumerate(truth) if t == c]
            if idx:
                out[c] = float(np.mean([pred[i] == c for i in idx]))
        return out

    per_class = []
    for c in labels:
        rs = [r for r in recs if r.actual_class == c]
        g = [r for r in rs if r.gtm_top]
        per_class.append({
            "class": c, "n": len(rs),
            "python_accuracy": acc([r.python_top for r in rs], [c] * len(rs)),
            "gtm_accuracy": acc([r.gtm_top for r in g], [c] * len(g)) if g else None,
            "final_accuracy": acc([r.final_category for r in rs], [c] * len(rs)),
            "agreement": float(np.mean([bool(r.class_match) for r in g])) if g else None,
            "mean_confidence_difference": float(np.mean([r.confidence_difference for r in g])) if g else None,
        })
    diffs = [r.confidence_difference for r in with_gtm if r.confidence_difference is not None]
    agree = [r for r in with_gtm if r.class_match]
    disagree = [r for r in with_gtm if not r.class_match]
    summary = {
        "n": len(recs),
        "n_with_gtm": len(with_gtm),
        "failed": (run.summary or {}).get("failed", []),
        "python": {"accuracy": acc(py, actual), "f1_macro": _f1(actual, py, labels),
                   "critical_recall": crit_recall(py, actual), "confusion_matrix": _cm(actual, py, labels)},
        "gtm": {"accuracy": acc([r.gtm_top for r in with_gtm], [r.actual_class for r in with_gtm]),
                "f1_macro": _f1([r.actual_class for r in with_gtm], [r.gtm_top for r in with_gtm], labels),
                "critical_recall": crit_recall([r.gtm_top for r in with_gtm], [r.actual_class for r in with_gtm]),
                "confusion_matrix": _cm([r.actual_class for r in with_gtm], [r.gtm_top for r in with_gtm], labels)},
        "final": {"accuracy": acc(final, actual), "f1_macro": _f1(actual, final, labels + [unknown_label()]),
                  "unknown": sum(1 for f in final if f == unknown_label())},
        "agreement_rate": (len(agree) / len(with_gtm)) if with_gtm else None,
        "both_correct": sum(1 for r in with_gtm if r.python_top == r.actual_class == r.gtm_top),
        "only_python_correct": sum(1 for r in with_gtm if r.python_top == r.actual_class != r.gtm_top),
        "only_gtm_correct": sum(1 for r in with_gtm if r.gtm_top == r.actual_class != r.python_top),
        "both_wrong": sum(1 for r in with_gtm if r.python_top != r.actual_class and r.gtm_top != r.actual_class),
        "mean_confidence_difference": float(np.mean(diffs)) if diffs else None,
        "mean_confidence_difference_agree": float(np.mean([r.confidence_difference for r in agree])) if agree else None,
        "mean_confidence_difference_disagree": float(np.mean([r.confidence_difference for r in disagree])) if disagree else None,
        "consistency": dict(Counter(r.consistency for r in recs)),
        "manual_review": sum(1 for r in recs if r.needs_review),
        "alerts": sum(1 for r in recs if r.alert_status == "Alert Generated"),
        "per_class": per_class,
        "top_confusions": Counter(f"{r.python_top} vs {r.gtm_top}" for r in disagree).most_common(8),
        "labels": labels,
        "python_version": run.python_version,
        "gtm_version": next((r.gtm_model_version for r in with_gtm if r.gtm_model_version), None),
    }
    summary["text"] = overall_text(summary)
    return summary


def overall_text(s: dict) -> str:
    if not s["n"]:
        return "No recordings evaluated yet."
    parts = [f"{s['n']} unseen test recordings were analysed."]
    p = s["python"]
    parts.append(f"Python model: accuracy {p['accuracy']:.1%}, macro F1 {p['f1_macro']:.3f}.")
    if s["n_with_gtm"]:
        g = s["gtm"]
        parts.append(f"GTM model: accuracy {g['accuracy']:.1%}, macro F1 {g['f1_macro']:.3f} on {s['n_with_gtm']} recordings.")
        parts.append(f"The models agreed on {s['agreement_rate']:.1%} of recordings; mean top-class confidence "
                     f"difference {s['mean_confidence_difference']:.3f}.")
        parts.append(f"Both correct: {s['both_correct']}, only Python: {s['only_python_correct']}, "
                     f"only GTM: {s['only_gtm_correct']}, both wrong: {s['both_wrong']}.")
    else:
        parts.append("No GTM results were available, so only the Python model was evaluated.")
    parts.append(f"Final decision accuracy {s['final']['accuracy']:.1%}; {s['manual_review']} recordings were routed "
                 f"to manual review.")
    return " ".join(parts)


def report_rows(run: EvaluationRun) -> tuple[list[str], list[list]]:
    labels = class_names()
    head = ["audio_id", "dataset_audio_id", "filename", "actual_class", "python_predicted_class"]
    head += [f"python_{c}" for c in labels]
    head += ["gtm_predicted_class"] + [f"gtm_{c}" for c in labels]
    head += ["class_match", "top_class_confidence_difference", "python_top_two_margin", "gtm_top_two_margin",
             "audio_quality", "severity", "alert_status", "manual_review", "review_reasons", "final_decision",
             "final_confidence", "python_correct", "gtm_correct", "final_correct", "explanation_of_disagreement"]
    rows = []
    for r in run_records(run):
        d = r.decision or {}
        ps, gs = r.python_scores or {}, r.gtm_scores or {}
        rows.append(
            [r.audio_id, r.dataset_audio_id, r.original_filename, r.actual_class, r.python_top]
            + [ps.get(c) for c in labels]
            + [r.gtm_top] + [gs.get(c) for c in labels]
            + [r.class_match, r.confidence_difference, (d.get("python") or {}).get("margin"),
               (d.get("gtm") or {}).get("margin"), r.quality_label, r.severity, r.alert_status, r.needs_review,
               "; ".join(r.review_reasons or []), r.final_category, r.final_confidence,
               r.python_top == r.actual_class, (r.gtm_top == r.actual_class) if r.gtm_top else None,
               r.final_category == r.actual_class, explain(r)])
    return head, rows


def finish(run: EvaluationRun) -> EvaluationRun:
    run.summary = {**(run.summary or {}), "result": compute_summary(run)}
    run.completed = len(run_records(run))
    run.status = "completed"
    run.finished_at = utcnow()
    audit.log("EVALUATION", "evaluation", run.run_code, {"completed": run.completed,
                                                          "python_accuracy": run.summary["result"]["python"]["accuracy"]})
    db.session.commit()
    return run
