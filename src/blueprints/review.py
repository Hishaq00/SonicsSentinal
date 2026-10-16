"""Manual review queue: uncertain and conflicting events."""
from __future__ import annotations

from collections import Counter

from flask import Blueprint, render_template, request

from ..models import AudioRecord
from ..security import permission_required
from ..services.review_service import REVIEW_REASONS

bp = Blueprint("review", __name__)

REASON_FILTERS = REVIEW_REASONS


@bp.route("/review")
@permission_required("review")
def queue():
    reason = request.args.get("reason") or ""
    q = AudioRecord.query.filter(AudioRecord.needs_review.is_(True), AudioRecord.is_evaluation.is_(False),
                                 AudioRecord.gtm_status != "pending")
    records = q.order_by(AudioRecord.created_at.desc()).limit(500).all()
    counts = Counter(r for rec in records for r in (rec.review_reasons or []))
    if reason:
        records = [r for r in records if any(reason in x for x in (r.review_reasons or []))]
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Informational": 4}
    sort = request.args.get("sort", "severity")
    if sort == "severity":
        records.sort(key=lambda r: (order.get(r.severity or "", 5), -(r.created_at.timestamp() if r.created_at else 0)))
    reviewed = (AudioRecord.query.filter(AudioRecord.reviewed_by.isnot(None), AudioRecord.is_evaluation.is_(False))
                .order_by(AudioRecord.reviewed_at.desc()).limit(10).all())
    return render_template("review/queue.html", records=records, counts=counts, reason=reason,
                           reasons=REASON_FILTERS, sort=sort, reviewed=reviewed)
