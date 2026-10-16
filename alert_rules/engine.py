"""Decision engine: model comparison, uncertainty, severity, alerts and review.

This module is pure Python (no database, no web framework) so every rule can be
unit-tested and explained. Inputs are the two models' class probabilities, the
audio-quality result and the detection history of the stream; the output is a
``Decision`` holding everything the SRS asks for in the final result:

* detected sound category, model agreement status, confidence level,
  event severity, alert status, recommended action, manual-review requirement,
* plus the evidence: confidence difference, top-two margins, overlapping
  classes, uncertainty reasons and the pass/fail state of every alert condition.

Order of evaluation
1. Summarise each model (top class, confidence, 2nd class, top-two margin, top 3).
2. Compare: class match, |Python top conf - GTM top conf|, margin difference.
3. Fuse the probability vectors (weighted mean) -> final candidate category.
3a. Live only: average the fused vector over the last ``live_smoothing_windows``
   windows. Live windows overlap (2 s every 1 s), so consecutive windows describe
   almost the same audio and a single window on its own is a noisy vote.
4. Consistency status: Acceptable Match / Weak Match / Model Disagreement /
   Uncertain Result (or Single-Model Result when GTM is unavailable).
5. Unknown handling: fused confidence below ``unknown_threshold`` or unusable
   audio -> "Unknown Sound".
6. Uncertainty reasons (low confidence, similar top classes, poor quality,
   disagreement, overlapping sounds, unsupported pattern).
7. Severity from the category rule, with rule-based escalation.
8. Alert conditions from the rule: confidence, top-two margin, model
   agreement, audio quality, repeated detection.
9. Manual-review routing and recommended action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from config import quality_rank, severity_rank, unknown_label

BACKGROUND = "Background Noise"

ALERT_GENERATED = "Alert Generated"
ALERT_PENDING = "Pending Confirmation"
ALERT_NOT_CONFIRMED = "Not Confirmed"
ALERT_SUPPRESSED = "Suppressed"
NO_ALERT = "No Alert"

REASON_TEXT = {
    "low_confidence": "Low confidence",
    "similar_top": "Similar top-class confidence scores",
    "poor_quality": "Poor audio quality",
    "disagreement": "Different predictions from the two models",
    "overlap": "Overlapping sounds",
    "unsupported": "Unsupported sound pattern",
    "critical_no_agreement": "Critical event without sufficient model agreement",
    "false_alarm": "Possible false alarm",
    "rule_always": "Category always requires manual review",
    "gtm_missing": "GTM prediction unavailable",
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    scores: dict[str, float]
    top: str
    confidence: float
    second: str
    second_confidence: float
    margin: float
    top3: list[tuple[str, float]]
    version: str | None = None


@dataclass
class HistoryItem:
    index: int
    time_s: float
    category: str
    confidence: float


@dataclass
class Context:
    source: str = "upload"  # "upload" or "live"
    time_s: float = 0.0
    index: int = 0
    history: list[HistoryItem] = field(default_factory=list)
    # live: fused scores of the windows just before this one (oldest first). The caller only
    # passes windows that belong to the same uninterrupted stretch of audio.
    recent_scores: list[dict[str, float]] = field(default_factory=list)
    n_segments: int = 1
    run_length: int | None = None  # uploads: length of the run of equal categories this segment belongs to
    repeat_count: int | None = None  # uploads: same-category detections within the repeat window
    escalation_count: int | None = None


@dataclass
class Decision:
    python: ModelResult
    gtm: ModelResult | None
    class_match: bool | None
    confidence_difference: float | None
    margin_difference: float | None
    consistency: str
    fused_scores: dict[str, float]  # what the decision is made on (live: smoothed over several windows)
    fused_margin: float
    window_scores: dict[str, float]  # this window's own fused scores, before smoothing
    smoothing_windows: int  # how many windows fused_scores averages over (1 = no smoothing)
    category: str
    confidence: float
    confidence_level: str
    overlapping: bool
    overlap_classes: list[str]
    uncertain: bool
    uncertain_reasons: list[str]
    severity: str
    escalated: bool
    escalation_note: str
    alert_enabled: bool
    alert_status: str
    alert_conditions: dict
    consecutive_count: int
    repeat_count: int
    needs_review: bool
    review_reasons: list[str]
    recommended_action: str
    responsible_roles: list[str]
    alert_type: str

    @property
    def event_status(self) -> str:
        if self.alert_status == ALERT_GENERATED:
            return "Alert Generated"
        if self.needs_review:
            return "Manual Review"
        if self.uncertain:
            return "Uncertain"
        return "Classified"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_status"] = self.event_status
        return d


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def summarize(scores: dict[str, float], version: str | None = None) -> ModelResult:
    """Top class, confidence, second class, top-two margin and top 3."""
    if not scores:
        raise ValueError("empty score dictionary")
    total = sum(max(0.0, float(v)) for v in scores.values())
    norm = {k: (max(0.0, float(v)) / total if total > 0 else 0.0) for k, v in scores.items()}
    ranked = sorted(norm.items(), key=lambda kv: kv[1], reverse=True)
    top, conf = ranked[0]
    second, second_conf = ranked[1] if len(ranked) > 1 else ("", 0.0)
    return ModelResult(
        scores=norm,
        top=top,
        confidence=conf,
        second=second,
        second_confidence=second_conf,
        margin=conf - second_conf,
        top3=[(k, round(v, 4)) for k, v in ranked[:3]],
        version=version,
    )


def confidence_level(conf: float, det: dict) -> str:
    if conf >= float(det.get("confidence_high", 0.8)):
        return "High"
    if conf >= float(det.get("confidence_medium", 0.6)):
        return "Medium"
    return "Low"


def max_severity(a: str, b: str) -> str:
    return a if severity_rank(a) >= severity_rank(b) else b


def fuse(py: ModelResult, gtm: ModelResult | None, weight_python: float) -> dict[str, float]:
    if gtm is None:
        return dict(py.scores)
    w_base = min(1.0, max(0.0, float(weight_python)))
    # When one model is decisive and the other is diffuse, scale relative weight by confidence power
    # so a clear human voice, alarm, gunshot or impact is not diluted by an uncertain guess.
    w_py = w_base * (max(0.01, float(py.confidence)) ** 1.5)
    w_gtm = (1.0 - w_base) * (max(0.01, float(gtm.confidence)) ** 1.5)
    total_w = w_py + w_gtm
    w = (w_py / total_w) if total_w > 0 else w_base
    classes = list(py.scores) + [c for c in gtm.scores if c not in py.scores]
    fused = {c: w * py.scores.get(c, 0.0) + (1.0 - w) * gtm.scores.get(c, 0.0) for c in classes}
    total = sum(fused.values()) or 1.0
    return {c: v / total for c, v in fused.items()}


def fusion_weight(det: dict, source: str) -> float:
    """Weight of the Python model in the fused score.

    The live path has its own value: there the Python model is trained on the
    microphone rendering of the dataset and is the more reliable of the two,
    so GTM corroborates rather than outvotes it.
    """
    base = float(det.get("fusion_weight_python", 0.5))
    if source == "live":
        return float(det.get("fusion_weight_python_live", base))
    return base


def smooth_live(window_scores: dict[str, float], ctx: Context, det: dict) -> tuple[dict[str, float], int]:
    """(scores to decide on, number of windows averaged).

    Live windows overlap by half, so a sound is present in two or more of them.
    Averaging the fused vector over the last few windows removes the flicker a
    single noisy window causes without hiding short events: a transient still
    dominates the average because it appears in every window that contains it.
    """
    n = int(det.get("live_smoothing_windows", 1) or 1)
    recent = [s for s in ctx.recent_scores if s][-(n - 1):] if n > 1 else []
    if ctx.source != "live" or not recent:
        return dict(window_scores), 1
    group = recent + [window_scores]
    classes = list(window_scores) + [c for s in recent for c in s if c not in window_scores]
    mean = {c: sum(s.get(c, 0.0) for s in group) / len(group) for c in dict.fromkeys(classes)}
    total = sum(mean.values()) or 1.0
    return {c: v / total for c, v in mean.items()}, len(group)


def consistency_status(py: ModelResult, gtm: ModelResult | None, quality: str, det: dict) -> str:
    if quality == "Unusable":
        return "Uncertain Result"
    if gtm is None:
        return "Single-Model Result"
    min_conf = float(det.get("min_confidence", 0.6))
    if py.top == gtm.top:
        diff = abs(py.confidence - gtm.confidence)
        if py.confidence >= min_conf and gtm.confidence >= min_conf and diff <= float(det.get("max_confidence_difference", 0.25)):
            return "Acceptable Match"
        return "Weak Match"
    if py.confidence >= min_conf and gtm.confidence >= min_conf:
        return "Model Disagreement"
    return "Uncertain Result"


def backward_counts(ctx: Context, category: str, window_s: float) -> tuple[int, int]:
    """(consecutive detections ending here, detections within ``window_s``) from the history."""
    consecutive = 1
    expected = ctx.index - 1
    for item in reversed(ctx.history):
        if item.index != expected or item.category != category:
            break
        consecutive += 1
        expected -= 1
    repeats = 1 + sum(1 for h in ctx.history if h.category == category and 0 <= ctx.time_s - h.time_s <= window_s)
    return consecutive, repeats


# --------------------------------------------------------------------------- #
# Main decision
# --------------------------------------------------------------------------- #
def decide(
    python_scores: dict[str, float],
    gtm_scores: dict[str, float] | None,
    quality_label: str,
    quality_metrics: dict | None,
    rules: dict[str, dict],
    detection: dict,
    ctx: Context | None = None,
    python_version: str | None = None,
    gtm_version: str | None = None,
) -> Decision:
    ctx = ctx or Context()
    det = detection
    unknown = unknown_label()
    quality_metrics = quality_metrics or {}

    py = summarize(python_scores, python_version)
    gtm = summarize(gtm_scores, gtm_version) if gtm_scores else None

    # 2. comparison
    class_match = (py.top == gtm.top) if gtm else None
    conf_diff = abs(py.confidence - gtm.confidence) if gtm else None
    margin_diff = abs(py.margin - gtm.margin) if gtm else None

    # 3-4. fusion, live smoothing and consistency
    window_scores = fuse(py, gtm, fusion_weight(det, ctx.source))
    fused, smoothing_windows = smooth_live(window_scores, ctx, det)
    f = summarize(fused)
    consistency = consistency_status(py, gtm, quality_label, det)

    # overlapping sounds: several (non-background) classes with significant confidence
    overlap_thr = float(det.get("overlap_threshold", 0.25))
    overlap_classes = [c for c, s in sorted(fused.items(), key=lambda kv: -kv[1]) if s >= overlap_thr and c != BACKGROUND]
    for model in (py, gtm):
        if model and model.second and model.second != BACKGROUND and model.second_confidence >= overlap_thr \
                and model.top != BACKGROUND and model.second not in overlap_classes:
            overlap_classes.append(model.second)
    overlapping = len(overlap_classes) >= 2

    # 5. category (Unknown handling)
    category, confidence = f.top, f.confidence
    is_unknown = confidence < float(det.get("unknown_threshold", 0.4)) or quality_label == "Unusable"
    if is_unknown:
        category = unknown
    level = confidence_level(confidence, det)

    # 6. uncertainty
    reasons: list[str] = []
    if confidence < float(det.get("min_confidence", 0.6)):
        reasons.append("low_confidence")
    if f.margin < float(det.get("top_two_margin", 0.15)):
        reasons.append("similar_top")
    if quality_label in ("Poor", "Unusable"):
        reasons.append("poor_quality")
    if gtm is not None and not class_match:
        reasons.append("disagreement")
    if overlapping:
        reasons.append("overlap")
    if is_unknown:
        reasons.append("unsupported")
    uncertain = bool(reasons)

    rule = rules.get(category) or rules.get(unknown) or {}
    esc = rule.get("escalation", {}) or {}

    # repetition statistics
    window = float(rule.get("repeat_window_seconds", 10))
    if ctx.source == "live":
        consecutive, repeats = backward_counts(ctx, category, window)
        _, esc_repeats = backward_counts(ctx, category, float(esc.get("within_seconds", window)))
    else:
        consecutive = ctx.run_length if ctx.run_length is not None else 1
        repeats = ctx.repeat_count if ctx.repeat_count is not None else consecutive
        esc_repeats = ctx.escalation_count if ctx.escalation_count is not None else repeats

    # 7. severity and escalation
    severity = rule.get("severity", "Low")
    escalated, escalation_note = False, ""
    if esc.get("enabled") and not is_unknown:
        if esc.get("condition") == "noise_level" and category == BACKGROUND and quality_metrics.get("noise_level_exceeded"):
            severity = max_severity(severity, esc.get("severity", "Medium"))
            escalated = True
            escalation_note = (f"Background noise level ({quality_metrics.get('noise_floor_dbfs')} dBFS) "
                               f"exceeds the configured limit.")
        elif esc.get("condition") == "repeated" and esc_repeats >= int(esc.get("min_repeats", 3)) \
                and confidence >= float(esc.get("min_confidence", 0.8)):
            severity = max_severity(severity, esc.get("severity", "High"))
            escalated = True
            escalation_note = (f"Detected {esc_repeats} times within {esc.get('within_seconds')} s "
                               f"with confidence >= {esc.get('min_confidence')}.")
        if escalated and esc.get("notify"):
            escalation_note += f" Notify: {esc['notify']}."

    # 8. alert conditions
    alert_enabled = bool(rule.get("alert_enabled")) or (escalated and bool(esc.get("generate_alert")))
    conditions: dict = {}
    alert_status = NO_ALERT
    if alert_enabled and not is_unknown:
        min_c = float(rule.get("min_confidence", det.get("min_confidence", 0.6)))
        min_m = float(rule.get("min_top_two_margin", det.get("top_two_margin", 0.15)))
        conditions["confidence"] = {"required": f">= {min_c:.2f}", "actual": round(confidence, 3), "passed": confidence >= min_c}
        conditions["top_two_margin"] = {"required": f">= {min_m:.2f}", "actual": round(f.margin, 3), "passed": f.margin >= min_m}
        need_agree = bool(rule.get("require_model_agreement"))
        if need_agree:
            if gtm is not None:
                agree_ok = bool(class_match)
                actual = "agree" if class_match else f"GTM says {gtm.top}"
                conditions["model_agreement"] = {"required": "both models agree", "actual": actual, "passed": agree_ok}
            else:
                # Single-model operation (when GTM is not configured): Python model drives the alert
                agree_ok = confidence >= min_c
                conditions["model_agreement"] = {
                    "required": f"Python confidence >= {min_c:.2f} (single-model mode)",
                    "actual": f"{confidence:.2f}",
                    "passed": agree_ok,
                }
        min_q = rule.get("min_audio_quality", "Poor")
        conditions["audio_quality"] = {"required": f"{min_q} or better", "actual": quality_label,
                                       "passed": quality_rank(quality_label) >= quality_rank(min_q)}
        need_rep = int(rule.get("required_consecutive_detections", 1))
        if need_rep > 1:
            if ctx.source != "live" and ctx.n_segments < need_rep:
                # Too short to repeat: demand agreement between models if GTM present, else require confidence
                agree_ok = bool(class_match) if gtm is not None else (confidence >= min_c)
                conditions["repeated_detection"] = {
                    "required": f"{need_rep} consecutive windows",
                    "actual": "recording too short; model agreement used instead" if gtm is not None else "recording too short; Python confidence verified",
                    "passed": agree_ok,
                }
            else:
                rep_ok = consecutive >= need_rep or repeats >= need_rep
                conditions["repeated_detection"] = {
                    "required": f"{need_rep} consecutive / within {int(window)} s",
                    "actual": f"{consecutive} consecutive, {repeats} in window",
                    "passed": rep_ok,
                }
        failed = [k for k, c in conditions.items() if not c["passed"]]
        if not failed:
            alert_status = ALERT_GENERATED
        elif failed == ["repeated_detection"]:
            alert_status = ALERT_PENDING if ctx.source == "live" else ALERT_NOT_CONFIRMED
        else:
            alert_status = ALERT_SUPPRESSED

    # 9. manual review
    review: list[str] = []
    cond = rule.get("manual_review_condition", "on_uncertain")
    if cond == "always":
        review.append("rule_always")
    if cond in ("on_uncertain", "on_uncertain_or_unconfirmed", "always") and uncertain:
        review.extend(reasons)
    if gtm is None and bool(rule.get("critical")) and alert_enabled:
        review.append("gtm_missing")
    if cond == "on_uncertain_or_unconfirmed":
        if alert_status == ALERT_SUPPRESSED and "model_agreement" in conditions and not conditions["model_agreement"]["passed"]:
            review.append("critical_no_agreement")
        elif alert_status in (ALERT_SUPPRESSED, ALERT_NOT_CONFIRMED):
            review.append("false_alarm")
    review = list(dict.fromkeys(review))  # de-duplicate, keep order
    needs_review = bool(review)

    action = rule.get("recommended_action", "")
    if alert_status == ALERT_PENDING:
        action = "Waiting for confirmation in the next window. " + action
    elif needs_review and alert_status != ALERT_GENERATED:
        action = "Manual review required: listen to the segment and confirm the category. " + (
            action if rule.get("critical") else "")
    if escalated and esc.get("notify") and alert_status == ALERT_GENERATED:
        action += f" Escalate to: {esc['notify']}."

    return Decision(
        python=py,
        gtm=gtm,
        class_match=class_match,
        confidence_difference=conf_diff,
        margin_difference=margin_diff,
        consistency=consistency,
        fused_scores=fused,
        fused_margin=f.margin,
        window_scores=window_scores,
        smoothing_windows=smoothing_windows,
        category=category,
        confidence=confidence,
        confidence_level=level,
        overlapping=overlapping,
        overlap_classes=overlap_classes if overlapping else [],
        uncertain=uncertain,
        uncertain_reasons=[REASON_TEXT[r] for r in reasons],
        severity=severity,
        escalated=escalated,
        escalation_note=escalation_note.strip(),
        alert_enabled=alert_enabled,
        alert_status=alert_status,
        alert_conditions=conditions,
        consecutive_count=consecutive,
        repeat_count=repeats,
        needs_review=needs_review,
        review_reasons=[REASON_TEXT[r] for r in review],
        recommended_action=action.strip(),
        responsible_roles=list(rule.get("responsible_roles", [])),
        alert_type=rule.get("alert_type", ""),
    )


# --------------------------------------------------------------------------- #
# Uploaded recordings: many segments
# --------------------------------------------------------------------------- #
@dataclass
class SegmentInput:
    index: int
    start: float
    end: float
    python_scores: dict[str, float] | None  # None = silent segment (not classified)
    gtm_scores: dict[str, float] | None
    quality_label: str
    quality_metrics: dict
    is_event: bool = True  # segment carries the event (within 15 dB of the loudest segment)


def _candidate(seg: SegmentInput, det: dict) -> str | None:
    if seg.python_scores is None:
        return None
    py = summarize(seg.python_scores)
    gtm = summarize(seg.gtm_scores) if seg.gtm_scores else None
    f = summarize(fuse(py, gtm, fusion_weight(det, "upload")))
    if f.confidence < float(det.get("unknown_threshold", 0.4)) or seg.quality_label == "Unusable":
        return unknown_label()
    return f.top


def decide_sequence(segments: list[SegmentInput], rules: dict, detection: dict,
                    python_version: str | None = None, gtm_version: str | None = None) -> list[Decision | None]:
    """Segment-level decisions for an uploaded recording.

    Pass 1 finds each segment's candidate category; pass 2 decides every
    segment knowing the length of the run of identical categories it belongs
    to, so an event seen in two overlapping segments counts as a repeated
    detection.
    """
    cands = [_candidate(s, detection) for s in segments]
    runs = [0] * len(segments)
    i = 0
    while i < len(segments):
        j = i
        while j + 1 < len(segments) and cands[j + 1] == cands[i] and cands[i] is not None:
            j += 1
        for k in range(i, j + 1):
            runs[k] = j - i + 1
        i = j + 1
    classified = sum(1 for c in cands if c is not None)
    out: list[Decision | None] = []
    for s, cand, run in zip(segments, cands, runs):
        if cand is None:
            out.append(None)
            continue
        rule = rules.get(cand, {})
        window = float(rule.get("repeat_window_seconds", 10))
        esc_window = float((rule.get("escalation") or {}).get("within_seconds", window))
        repeat = sum(1 for o, c in zip(segments, cands) if c == cand and abs(o.start - s.start) <= window)
        esc_count = sum(1 for o, c in zip(segments, cands) if c == cand and abs(o.start - s.start) <= esc_window)
        ctx = Context(source="upload", time_s=s.start, index=s.index, n_segments=classified,
                      run_length=run, repeat_count=repeat, escalation_count=esc_count)
        out.append(decide(s.python_scores, s.gtm_scores, s.quality_label, s.quality_metrics, rules, detection,
                          ctx, python_version, gtm_version))
    return out


def mean_scores(score_list: list[dict[str, float]]) -> dict[str, float] | None:
    score_list = [s for s in score_list if s]
    if not score_list:
        return None
    keys = list(score_list[0])
    return {k: sum(s.get(k, 0.0) for s in score_list) / len(score_list) for k in keys}


def merge_events(segments: list[SegmentInput], decisions: list[Decision | None]) -> list[dict]:
    """Merge consecutive segments with the same final category into timeline events."""
    events: list[dict] = []
    for seg, d in zip(segments, decisions):
        if d is None:
            continue
        if events and events[-1]["category"] == d.category and seg.start <= events[-1]["end"] + 0.01:
            ev = events[-1]
            ev["end"] = seg.end
            ev["segments"].append(seg.index)
            ev["confidence"] = max(ev["confidence"], d.confidence)
            ev["severity"] = max_severity(ev["severity"], d.severity)
            ev["alert"] = ev["alert"] or d.alert_status == ALERT_GENERATED
        else:
            events.append({
                "category": d.category,
                "start": seg.start,
                "end": seg.end,
                "segments": [seg.index],
                "confidence": d.confidence,
                "severity": d.severity,
                "alert": d.alert_status == ALERT_GENERATED,
            })
    for ev in events:
        ev["confidence"] = round(ev["confidence"], 4)
    return events


def decide_recording(segments: list[SegmentInput], rules: dict, detection: dict, quality_label: str,
                     quality_metrics: dict, python_version: str | None = None, gtm_version: str | None = None) -> dict:
    """Final decision for a whole uploaded recording.

    The recording-level model outputs are the mean probabilities over the
    event-bearing segments (the same aggregation used to evaluate the models).
    Repetition is the longest run of segments whose own category equals the
    recording category. Segment decisions build the event timeline and can
    raise extra alerts for a second critical event in long recordings.
    """
    seg_decisions = decide_sequence(segments, rules, detection, python_version, gtm_version)
    classified = [(s, d) for s, d in zip(segments, seg_decisions) if d is not None]
    if not classified:
        return {"decision": None, "segment_decisions": seg_decisions, "events": [], "extra_alerts": [],
                "silent": True}
    event_segs = [s for s, _ in classified if s.is_event] or [s for s, _ in classified]
    py_rec = mean_scores([s.python_scores for s in event_segs])
    gtm_list = [s.gtm_scores for s in event_segs if s.gtm_scores]
    gtm_rec = mean_scores(gtm_list) if len(gtm_list) == len(event_segs) else None

    provisional = summarize(fuse(summarize(py_rec), summarize(gtm_rec) if gtm_rec else None,
                                 fusion_weight(detection, "upload")))
    category = provisional.top
    best_run = 0
    run = 0
    for s, d in classified:
        run = run + 1 if d.category == category else 0
        best_run = max(best_run, run)
    rule = rules.get(category, {})
    window = float(rule.get("repeat_window_seconds", 10))
    times = [s.start for s, d in classified if d.category == category]
    repeat = max((sum(1 for u in times if 0 <= u - t <= window) for t in times), default=0)
    esc_window = float((rule.get("escalation") or {}).get("within_seconds", window))
    esc_count = max((sum(1 for u in times if 0 <= u - t <= esc_window) for t in times), default=0)
    ctx = Context(source="upload", n_segments=len(classified), run_length=max(best_run, 1),
                  repeat_count=max(repeat, 1), escalation_count=max(esc_count, 1))
    decision = decide(py_rec, gtm_rec, quality_label, quality_metrics, rules, detection, ctx,
                      python_version, gtm_version)

    # Additional alerts: a *different* critical event confirmed at segment level.
    extra_alerts = []
    seen = {decision.category}
    for s, d in classified:
        if d.alert_status == ALERT_GENERATED and d.category not in seen:
            seen.add(d.category)
            extra_alerts.append({"segment_index": s.index, "decision": d})
    return {
        "decision": decision,
        "segment_decisions": seg_decisions,
        "events": merge_events(segments, seg_decisions),
        "extra_alerts": extra_alerts,
        "silent": False,
        "python_recording_scores": py_rec,
        "gtm_recording_scores": gtm_rec,
    }
