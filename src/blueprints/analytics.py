"""Analytics and reporting page (charts are drawn from /api/analytics)."""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request
from flask_login import current_user

from config import class_info

from ..security import permission_required
from ..services import analytics

bp = Blueprint("analytics", __name__)


def _days() -> int:
    return min(365, max(1, request.args.get("days", 30, type=int)))


@bp.route("/analytics")
@permission_required("analytics")
def index():
    days = _days()
    return render_template(
        "analytics/index.html", days=days,
        kpis=analytics.kpis(current_user, days),
        fpfn=analytics.false_positive_negative(current_user, max(days, 30)),
        response=analytics.alert_response(current_user, days),
        pairs=analytics.disagreement_pairs(current_user, days),
        consistency=analytics.consistency_counts(current_user, days),
    )


@bp.route("/api/analytics")
@permission_required("analytics")
def data():
    days = _days()
    info = class_info()
    cats = analytics.category_counts(current_user, days)
    return jsonify(
        days=days,
        categories=[{**c, "color": info.get(c["category"], {}).get("color", "#8a857a")} for c in cats],
        trend=analytics.daily_trend(current_user, min(days, 60)),
        confidence=analytics.confidence_distribution(current_user, days),
        quality=analytics.quality_distribution(current_user, days),
        consistency=analytics.consistency_counts(current_user, days),
        scatter=analytics.python_vs_gtm_confidence(current_user, days),
        response=analytics.alert_response(current_user, days),
        fpfn=analytics.false_positive_negative(current_user, max(days, 30)),
    )
