"""Queries for dashboards, event history filters and analytics."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from statistics import median

from sqlalchemy import and_, func, or_

from config import SEVERITY_LEVELS, class_names

from ..extensions import db
from ..models import Alert, AlertAction, AudioRecord, Review, User, utcnow
from ..security import can

BACKGROUND = "Background Noise"


def scoped_records(user):
    q = AudioRecord.query.filter(AudioRecord.is_evaluation.is_(False))
    if not can(user, "view_all_records"):
        q = q.filter(AudioRecord.user_id == user.id)
    return q


FILTER_KEYS = ("audio_id", "filename", "category", "date_from", "date_to", "conf_min", "conf_max", "severity",
               "quality", "review", "status", "user", "source", "consistency", "include_background")


def _parse_date(text: str, end: bool = False) -> datetime | None:
    try:
        d = datetime.strptime(text, "%Y-%m-%d")
        return d + timedelta(days=1) if end else d
    except (TypeError, ValueError):
        return None


def apply_filters(q, f: dict):
    """Event-history search and filtering (FR lxvii)."""
    if f.get("audio_id"):
        q = q.filter(AudioRecord.audio_id.ilike(f"%{f['audio_id'].strip()}%"))
    if f.get("filename"):
        q = q.filter(AudioRecord.original_filename.ilike(f"%{f['filename'].strip()}%"))
    if f.get("category"):
        cat = f["category"]
        q = q.filter(or_(AudioRecord.reviewed_category == cat,
                         and_(AudioRecord.reviewed_category.is_(None), AudioRecord.final_category == cat)))
    if _parse_date(f.get("date_from")):
        q = q.filter(AudioRecord.created_at >= _parse_date(f["date_from"]))
    if _parse_date(f.get("date_to")):
        q = q.filter(AudioRecord.created_at < _parse_date(f["date_to"], end=True))
    for key, op in (("conf_min", ">="), ("conf_max", "<=")):
        try:
            v = float(f.get(key)) / (100.0 if float(f.get(key)) > 1 else 1.0)
        except (TypeError, ValueError):
            continue
        q = q.filter(AudioRecord.final_confidence >= v if op == ">=" else AudioRecord.final_confidence <= v)
    if f.get("severity"):
        q = q.filter(or_(AudioRecord.reviewed_severity == f["severity"],
                         and_(AudioRecord.reviewed_severity.is_(None), AudioRecord.severity == f["severity"])))
    if f.get("quality"):
        q = q.filter(AudioRecord.quality_label == f["quality"])
    if f.get("review") == "pending":
        q = q.filter(AudioRecord.needs_review.is_(True))
    elif f.get("review") == "reviewed":
        q = q.filter(AudioRecord.reviewed_by.isnot(None))
    elif f.get("review") == "not_required":
        q = q.filter(AudioRecord.needs_review.is_(False), AudioRecord.reviewed_by.is_(None))
    if f.get("status"):
        q = q.filter(AudioRecord.status == f["status"])
    if f.get("consistency"):
        q = q.filter(AudioRecord.consistency == f["consistency"])
    if f.get("source"):
        if f["source"] == "upload":
            q = q.filter(AudioRecord.source.in_(("upload", "batch")))
        else:
            q = q.filter(AudioRecord.source == f["source"])
    if f.get("user"):
        q = q.join(User, AudioRecord.user_id == User.id).filter(User.username.ilike(f"%{f['user'].strip()}%"))
    if not f.get("include_background") and not f.get("category"):
        q = q.filter(or_(AudioRecord.source != "live", AudioRecord.final_category != BACKGROUND,
                         AudioRecord.final_category.is_(None)))
    return q


def kpis(user, days: int = 30) -> dict:
    since = utcnow() - timedelta(days=days)
    base = scoped_records(user).filter(AudioRecord.created_at >= since, AudioRecord.final_category.isnot(None))
    total = base.count()
    events = base.filter(AudioRecord.final_category != BACKGROUND).count()
    avg_conf = base.with_entities(func.avg(AudioRecord.final_confidence)).scalar()
    compared = base.filter(AudioRecord.class_match.isnot(None))
    n_compared = compared.count()
    disagree = compared.filter(AudioRecord.class_match.is_(False)).count()
    poor = base.filter(AudioRecord.quality_label.in_(("Poor", "Unusable"))).count()
    review = scoped_records(user).filter(AudioRecord.needs_review.is_(True)).count()
    alert_q = Alert.query.join(AudioRecord).filter(AudioRecord.is_evaluation.is_(False))
    if not can(user, "view_all_records"):
        alert_q = alert_q.filter(AudioRecord.user_id == user.id)
    active_alerts = alert_q.filter(Alert.status.in_(("Active", "Escalated"))).count()
    critical = alert_q.filter(Alert.severity == "Critical", Alert.created_at >= since).count()
    return {
        "total": total,
        "events": events,
        "avg_confidence": float(avg_conf) if avg_conf is not None else None,
        "compared": n_compared,
        "disagreements": disagree,
        "disagreement_rate": (disagree / n_compared) if n_compared else None,
        "poor_quality": poor,
        "pending_review": review,
        "active_alerts": active_alerts,
        "critical_alerts": critical,
        "days": days,
    }


def category_counts(user, days: int = 30) -> list[dict]:
    since = utcnow() - timedelta(days=days)
    cat = func.coalesce(AudioRecord.reviewed_category, AudioRecord.final_category)
    rows = (scoped_records(user).filter(AudioRecord.created_at >= since, AudioRecord.final_category.isnot(None))
            .with_entities(cat, func.count()).group_by(cat).all())
    counts = dict(rows)
    order = class_names() + [c for c in counts if c not in class_names()]
    return [{"category": c, "count": int(counts.get(c, 0))} for c in order if c in counts or c in class_names()]


def daily_trend(user, days: int = 14) -> dict:
    start = (utcnow() - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    day = func.date(AudioRecord.created_at)
    rows = (scoped_records(user).filter(AudioRecord.created_at >= start, AudioRecord.final_category.isnot(None),
                                        AudioRecord.final_category != BACKGROUND)
            .with_entities(day, AudioRecord.severity, func.count()).group_by(day, AudioRecord.severity).all())
    labels = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    series = {sev: [0] * days for sev in SEVERITY_LEVELS}
    for d, sev, n in rows:
        key = str(d)[:10]
        if key in labels and sev in series:
            series[sev][labels.index(key)] += int(n)
    return {"labels": labels, "series": series}


def confidence_distribution(user, days: int = 30, bins: int = 10) -> dict:
    since = utcnow() - timedelta(days=days)
    rows = (scoped_records(user).filter(AudioRecord.created_at >= since, AudioRecord.final_confidence.isnot(None))
            .with_entities(AudioRecord.final_confidence, AudioRecord.python_confidence, AudioRecord.gtm_confidence).all())
    edges = [i / bins for i in range(bins + 1)]

    def hist(values):
        h = [0] * bins
        for v in values:
            if v is None:
                continue
            h[min(bins - 1, int(float(v) * bins))] += 1
        return h

    return {
        "labels": [f"{int(edges[i] * 100)}-{int(edges[i + 1] * 100)}%" for i in range(bins)],
        "final": hist(r[0] for r in rows),
        "python": hist(r[1] for r in rows),
        "gtm": hist(r[2] for r in rows),
    }


def quality_distribution(user, days: int = 30) -> dict:
    since = utcnow() - timedelta(days=days)
    rows = (scoped_records(user).filter(AudioRecord.created_at >= since)
            .with_entities(AudioRecord.quality_label, func.count()).group_by(AudioRecord.quality_label).all())
    counts = {k or "Unknown": int(v) for k, v in rows}
    return {q: counts.get(q, 0) for q in ("Good", "Acceptable", "Poor", "Unusable")}


def disagreement_pairs(user, days: int = 30, limit: int = 8) -> list[dict]:
    since = utcnow() - timedelta(days=days)
    rows = (scoped_records(user).filter(AudioRecord.created_at >= since, AudioRecord.class_match.is_(False))
            .with_entities(AudioRecord.python_top, AudioRecord.gtm_top, func.count())
            .group_by(AudioRecord.python_top, AudioRecord.gtm_top).order_by(func.count().desc()).limit(limit).all())
    return [{"python": p, "gtm": g, "count": int(n)} for p, g, n in rows]


def consistency_counts(user, days: int = 30) -> dict:
    since = utcnow() - timedelta(days=days)
    rows = (scoped_records(user).filter(AudioRecord.created_at >= since, AudioRecord.consistency.isnot(None))
            .with_entities(AudioRecord.consistency, func.count()).group_by(AudioRecord.consistency).all())
    return {k: int(v) for k, v in rows}


def false_positive_negative(user, days: int = 90) -> dict:
    """Derived from reviewer decisions: model said X but reviewer said Y."""
    since = utcnow() - timedelta(days=days)
    q = Review.query.join(AudioRecord).filter(Review.created_at >= since)
    if not can(user, "view_all_records"):
        q = q.filter(AudioRecord.user_id == user.id)
    fp, fn, confirmed = Counter(), Counter(), 0
    total = 0
    latest: dict[int, Review] = {}
    for r in q.order_by(Review.created_at).all():
        latest[r.record_id] = r
    for r in latest.values():
        total += 1
        if r.original_category == r.reviewed_category:
            confirmed += 1
            continue
        if r.original_category:
            fp[r.original_category] += 1
        if r.reviewed_category:
            fn[r.reviewed_category] += 1
    cats = class_names()
    return {
        "reviewed": total,
        "confirmed": confirmed,
        "precision_of_reviewed": (confirmed / total) if total else None,
        "rows": [{"category": c, "false_positives": fp.get(c, 0), "false_negatives": fn.get(c, 0)} for c in cats],
    }


def alert_response(user, days: int = 30) -> dict:
    since = utcnow() - timedelta(days=days)
    q = Alert.query.join(AudioRecord).filter(Alert.created_at >= since, AudioRecord.is_evaluation.is_(False))
    if not can(user, "view_all_records"):
        q = q.filter(AudioRecord.user_id == user.id)
    alerts = q.all()
    by_sev: dict[str, list[float]] = defaultdict(list)
    counts = Counter()
    status = Counter()
    for a in alerts:
        counts[a.severity] += 1
        status[a.status] += 1
        if a.response_seconds is not None:
            by_sev[a.severity].append(a.response_seconds)
    actions = Counter(dict(
        db.session.query(AlertAction.action, func.count()).join(Alert)
        .filter(Alert.created_at >= since).group_by(AlertAction.action).all()))
    return {
        "total": len(alerts),
        "by_status": dict(status),
        "actions": dict(actions),
        "severity": [{"severity": s, "count": counts.get(s, 0),
                      "responded": len(by_sev.get(s, [])),
                      "median_response_s": median(by_sev[s]) if by_sev.get(s) else None,
                      "mean_response_s": (sum(by_sev[s]) / len(by_sev[s])) if by_sev.get(s) else None}
                     for s in reversed(SEVERITY_LEVELS)],
    }


def critical_timeline(user, hours: int = 24) -> list[dict]:
    since = utcnow() - timedelta(hours=hours)
    q = Alert.query.join(AudioRecord).filter(Alert.created_at >= since,
                                             Alert.severity.in_(("Critical", "High")), AudioRecord.is_evaluation.is_(False))
    if not can(user, "view_all_records"):
        q = q.filter(AudioRecord.user_id == user.id)
    return [{"code": a.alert_code, "category": a.category, "severity": a.severity, "status": a.status,
             "created_at": a.created_at, "audio_id": a.record.audio_id} for a in q.order_by(Alert.created_at).all()]


def hourly_activity(user, hours: int = 24) -> dict:
    """Events per hour for the last ``hours``, split into event and critical counts.

    Drawn as a strip chart, so the console reads like a recorder trace rather than
    a bar chart: one value per hour, oldest first.
    """
    start = (utcnow() - timedelta(hours=hours - 1)).replace(minute=0, second=0, microsecond=0)
    rows = (scoped_records(user).filter(AudioRecord.created_at >= start,
                                        AudioRecord.final_category.isnot(None))
            .with_entities(AudioRecord.created_at, AudioRecord.final_category, AudioRecord.severity).all())
    events = [0] * hours
    critical = [0] * hours
    background = [0] * hours
    for created, category, severity in rows:
        i = int((created - start).total_seconds() // 3600)
        if not 0 <= i < hours:
            continue
        if category == BACKGROUND:
            background[i] += 1
        else:
            events[i] += 1
            if severity in ("Critical", "High"):
                critical[i] += 1
    return {"labels": [(start + timedelta(hours=i)).strftime("%H:%M") for i in range(hours)],
            "events": events, "critical": critical, "background": background,
            "start": start.isoformat(timespec="minutes")}


def recent_traces(user, limit: int = 6) -> list[dict]:
    """The newest classified recordings that still have their waveform on disk.

    The console draws the real trace of each one, so the page shows the audio that
    was actually analysed instead of a decorative graphic.
    """
    from . import storage

    rows = (scoped_records(user).filter(AudioRecord.final_category.isnot(None),
                                        AudioRecord.waveform_path.isnot(None),
                                        AudioRecord.audio_deleted.is_(False))
            .order_by(AudioRecord.created_at.desc()).limit(limit * 3).all())
    out: list[dict] = []
    for r in rows:
        path = storage.resolve(r.waveform_path)
        if not path or not path.exists():
            continue
        try:
            import json

            peaks = json.loads(path.read_text(encoding="utf-8")).get("peaks") or []
        except (OSError, ValueError):
            continue
        if not peaks:
            continue
        # One value per column is enough for a 200 px trace.
        step = max(1, len(peaks) // 180)
        out.append({
            "audio_id": r.audio_id,
            "category": r.final_category,
            "confidence": round(r.final_confidence or 0, 3),
            "severity": r.severity or "Low",
            "source": r.source,
            "duration": r.duration,
            "created_at": r.created_at.isoformat(timespec="seconds") if r.created_at else None,
            "peaks": [round(max(abs(p[0]), abs(p[1])), 3) for p in peaks[::step][:180]],
        })
        if len(out) >= limit:
            break
    return out


def python_vs_gtm_confidence(user, days: int = 30, limit: int = 400) -> list[list[float]]:
    since = utcnow() - timedelta(days=days)
    rows = (scoped_records(user).filter(AudioRecord.created_at >= since, AudioRecord.gtm_confidence.isnot(None))
            .with_entities(AudioRecord.python_confidence, AudioRecord.gtm_confidence, AudioRecord.class_match)
            .order_by(AudioRecord.created_at.desc()).limit(limit).all())
    return [[round(p or 0, 3), round(g or 0, 3), bool(m)] for p, g, m in rows]
