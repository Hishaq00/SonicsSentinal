"""Event history: search, filtering and CSV / Excel export."""
from __future__ import annotations

from flask import Blueprint, Response, render_template, request

from config import CONSISTENCY_STATUSES, EVENT_STATUSES, QUALITY_LEVELS, SEVERITY_LEVELS, class_names, unknown_label
from flask_login import current_user, login_required
from python_models.inference import get_classifier

from ..extensions import db
from ..models import AudioRecord
from ..security import can, permission_required
from ..services import analytics, audit, reports

bp = Blueprint("history", __name__)


def _filters() -> dict:
    f = {k: (request.args.get(k) or "").strip() for k in analytics.FILTER_KEYS}
    if not can(current_user, "view_all_records"):
        f["user"] = ""
    return f


@bp.route("/history")
@login_required
def history():
    f = _filters()
    q = analytics.apply_filters(analytics.scoped_records(current_user), f)
    page = max(1, request.args.get("page", 1, type=int))
    per_page = min(100, max(10, request.args.get("per_page", 25, type=int)))
    pagination = q.order_by(AudioRecord.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template(
        "history/list.html", records=pagination.items, pagination=pagination, f=f,
        classes=class_names() + [unknown_label()], severities=SEVERITY_LEVELS, qualities=QUALITY_LEVELS,
        statuses=EVENT_STATUSES, consistencies=CONSISTENCY_STATUSES,
        show_user=can(current_user, "view_all_records"), can_export=can(current_user, "export"),
    )


@bp.route("/history/export.<fmt>")
@permission_required("export")
def export(fmt):
    if fmt not in ("csv", "xlsx"):
        return Response("Unsupported format", status=400)
    f = _filters()
    q = analytics.apply_filters(analytics.scoped_records(current_user), f)
    records = q.order_by(AudioRecord.created_at.desc()).limit(50000).all()
    clf = get_classifier()
    classes = clf.classes if clf.available else class_names()
    head, rows = reports.record_rows(records, classes)
    audit.log("EXPORT", "records", None, {"format": fmt, "rows": len(rows),
                                          "filters": {k: v for k, v in f.items() if v}})
    db.session.commit()
    name = f"sonicsentinel_events.{fmt}"
    if fmt == "csv":
        return Response(reports.to_csv(head, rows), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={name}"})
    data = reports.to_xlsx([("Events", head, rows)])
    return Response(data, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename={name}"})
