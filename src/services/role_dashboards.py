"""Queries for the role-specific dashboards.

* Audio reviewer      -> review desk (queue workload, priorities, corrections)
* Security operator   -> security console  } alerts and detections for the
* Maintenance operator -> maintenance console } categories the role is responsible for

Which categories an operator watches is not hard-coded: it comes from the
``responsible_roles`` of each rule in ``alert_rules/alert_rules.json``, so an
administrator who reassigns a category on the Alert rules page changes the
console too.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
from statistics import median

from sqlalchemy import func

from alert_rules import rules_by_category
from config import SEVERITY_LEVELS, class_info

from ..models import Alert, AlertAction, AudioRecord, Review, utcnow
from . import analytics
from .alerts_service import ACTIVE_STATES, visible_alerts
from .review_service import REVIEW_REASONS

SEVERITY_RANK = {s: i for i, s in enumerate(reversed(SEVERITY_LEVELS))}  # Critical = 0
EFFECTIVE_CATEGORY = func.coalesce(AudioRecord.reviewed_category, AudioRecord.final_category)


def _sev_rank(severity: str | None) -> int:
    return SEVERITY_RANK.get(severity or "", len(SEVERITY_RANK))


def _start_of_today():
    return utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


# --------------------------------------------------------------------------- #
# Audio reviewer
# --------------------------------------------------------------------------- #
def review_queue_query():
    """Records waiting for a reviewer (same definition as the Review queue page)."""
    return AudioRecord.query.filter(AudioRecord.needs_review.is_(True), AudioRecord.is_evaluation.is_(False),
                                    AudioRecord.gtm_status != "pending")


def reviewer_dashboard(user, days: int = 30) -> dict:
    now = utcnow()
    queue = review_queue_query().all()
    reason_counts = Counter()
    for rec in queue:
        for reason in REVIEW_REASONS:
            if any(reason in r for r in (rec.review_reasons or [])):
                reason_counts[reason] += 1
    # Most urgent first: severity, then the one that has waited longest.
    priority = sorted(queue, key=lambda r: (_sev_rank(r.severity), r.created_at or now))[:12]
    oldest = min((r.created_at for r in queue if r.created_at), default=None)

    since = now - timedelta(days=days)
    week = now - timedelta(days=7)
    team = Review.query.filter(Review.created_at >= since).all()
    mine_week = [r for r in team if r.reviewer_id == user.id and r.created_at >= week]
    corrections = Counter((r.original_category or "-", r.reviewed_category or "-")
                          for r in team if r.decision != "confirmed")
    return {
        "open": len(queue),
        "urgent": sum(1 for r in queue if r.severity in ("Critical", "High")),
        "disagreements": sum(1 for r in queue if r.class_match is False),
        "oldest_wait_s": (now - oldest).total_seconds() if oldest else None,
        "priority": priority,
        "reasons": [{"reason": r, "count": reason_counts.get(r, 0)} for r in REVIEW_REASONS],
        "my_reviews_week": len(mine_week),
        "my_overrides_week": sum(1 for r in mine_week if r.is_override),
        "my_today": sum(1 for r in mine_week if r.created_at >= _start_of_today()),
        "team_reviews": len(team),
        "override_rate": (sum(1 for r in team if r.is_override) / len(team)) if team else None,
        "false_alarms": sum(1 for r in team if r.decision == "false_alarm"),
        "corrections": [{"model": a, "reviewer": b, "count": n} for (a, b), n in corrections.most_common(8)],
        "my_recent": (Review.query.filter(Review.reviewer_id == user.id)
                      .order_by(Review.created_at.desc()).limit(8).all()),
        "pairs": analytics.disagreement_pairs(user, days),
        "days": days,
    }


# --------------------------------------------------------------------------- #
# Security / maintenance operator
# --------------------------------------------------------------------------- #
def watched_rules(role: str) -> list[dict]:
    """Alert rules the role is responsible for, most severe first."""
    rules = [r for r in rules_by_category().values() if role in (r.get("responsible_roles") or [])]
    return sorted(rules, key=lambda r: (not r.get("alert_enabled"), _sev_rank(r.get("severity")), r["category"]))


def operator_dashboard(user, trend_days: int = 14) -> dict:
    now = utcnow()
    rules = watched_rules(user.role)
    cats = [r["category"] for r in rules]
    priority_cats = [r["category"] for r in rules if r.get("alert_enabled") or r.get("critical")]

    alerts_q = visible_alerts(user).filter(Alert.category.in_(cats))
    open_alerts = alerts_q.filter(Alert.status.in_(ACTIVE_STATES)).all()
    open_alerts.sort(key=lambda a: (_sev_rank(a.severity), a.status != "Active", -(a.created_at.timestamp())))

    month = alerts_q.filter(Alert.created_at >= now - timedelta(days=30)).all()
    responses = [a.response_seconds for a in month if a.response_seconds is not None]

    records = analytics.scoped_records(user).filter(EFFECTIVE_CATEGORY.in_(cats))
    day_ago, week_ago = now - timedelta(hours=24), now - timedelta(days=7)
    counts_24h = dict(records.filter(AudioRecord.created_at >= day_ago)
                      .with_entities(EFFECTIVE_CATEGORY, func.count()).group_by(EFFECTIVE_CATEGORY).all())
    counts_7d = dict(records.filter(AudioRecord.created_at >= week_ago)
                     .with_entities(EFFECTIVE_CATEGORY, func.count()).group_by(EFFECTIVE_CATEGORY).all())
    last_seen = dict(records.with_entities(EFFECTIVE_CATEGORY, func.max(AudioRecord.created_at))
                     .group_by(EFFECTIVE_CATEGORY).all())
    open_by_cat = Counter(a.category for a in open_alerts)

    categories = [{
        "rule": r,
        "category": r["category"],
        "events_24h": int(counts_24h.get(r["category"], 0)),
        "events_7d": int(counts_7d.get(r["category"], 0)),
        "open_alerts": open_by_cat.get(r["category"], 0),
        "last_seen": last_seen.get(r["category"]),
        "priority": r["category"] in priority_cats,
    } for r in rules]

    start = (now - timedelta(days=trend_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    day = func.date(AudioRecord.created_at)
    trend_rows = (records.filter(AudioRecord.created_at >= start, EFFECTIVE_CATEGORY.in_(priority_cats))
                  .with_entities(day, EFFECTIVE_CATEGORY, func.count()).group_by(day, EFFECTIVE_CATEGORY).all())
    days = [(start + timedelta(days=i)).date().isoformat() for i in range(trend_days)]
    series = {c: [0] * trend_days for c in priority_cats}
    for d, c, n in trend_rows:
        key = d if isinstance(d, str) else d.isoformat()
        if key in days and c in series:
            series[c][days.index(key)] = int(n)

    my_actions = (AlertAction.query.filter(AlertAction.user_id == user.id)
                  .order_by(AlertAction.created_at.desc()).limit(8).all())
    return {
        "rules": rules,
        "categories": categories,
        "category_names": cats,
        "priority_names": priority_cats,
        "open_alerts": open_alerts[:15],
        "open_total": len(open_alerts),
        "unacknowledged": sum(1 for a in open_alerts if a.status == "Active"),
        "critical_open": sum(1 for a in open_alerts if a.severity == "Critical"),
        "escalated_open": sum(1 for a in open_alerts if a.status == "Escalated"),
        "events_24h": sum(counts_24h.values()),
        "median_response_s": median(responses) if responses else None,
        "alerts_30d": len(month),
        "my_actions_today": AlertAction.query.filter(AlertAction.user_id == user.id,
                                                     AlertAction.created_at >= _start_of_today()).count(),
        "my_actions": my_actions,
        "timeline": [t for t in analytics.critical_timeline(user, 24) if t["category"] in cats],
        "recent": (records.filter(EFFECTIVE_CATEGORY.in_(priority_cats))
                   .order_by(AudioRecord.created_at.desc()).limit(10).all()),
        "trend": {"days": days, "series": series,
                  "colors": {c: class_info().get(c, {}).get("color", "#8a857a") for c in priority_cats}},
    }
