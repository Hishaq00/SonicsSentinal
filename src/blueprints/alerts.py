"""Alert list, alert actions (acknowledge / dismiss / escalate / resolve) and the
real-time alert feed polled by every page."""
from __future__ import annotations

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from config import SEVERITY_LEVELS, class_names

from ..extensions import db
from ..models import Alert
from ..security import can_act_on_alert
from ..services import alerts_service
from ..utils import iso

bp = Blueprint("alerts", __name__)
TABS = {
    "active": ("Active", "Escalated"),
    "acknowledged": ("Acknowledged",),
    "closed": ("Resolved", "Dismissed"),
    "all": None,
}
ESCALATION_TARGETS = ["Security lead", "Emergency services (manual call)", "Maintenance lead", "Facility manager",
                      "Site supervisor"]


@bp.route("/alerts")
@login_required
def list_alerts():
    tab = request.args.get("tab", "active")
    if tab not in TABS:
        tab = "active"
    q = alerts_service.visible_alerts(current_user)
    statuses = TABS[tab]
    if statuses:
        q = q.filter(Alert.status.in_(statuses))
    category = request.args.get("category") or ""
    severity = request.args.get("severity") or ""
    if category:
        q = q.filter(Alert.category == category)
    if severity:
        q = q.filter(Alert.severity == severity)
    page = max(1, request.args.get("page", 1, type=int))
    pagination = q.order_by(Alert.created_at.desc()).paginate(page=page, per_page=25, error_out=False)
    counts = {}
    base = alerts_service.visible_alerts(current_user)
    for key, st in TABS.items():
        qq = base if st is None else base.filter(Alert.status.in_(st))
        counts[key] = qq.count()
    return render_template("alerts/list.html", alerts=pagination.items, pagination=pagination, tab=tab,
                           counts=counts, category=category, severity=severity, classes=class_names(),
                           severities=SEVERITY_LEVELS, targets=ESCALATION_TARGETS, can_act=can_act_on_alert)


@bp.route("/alerts/<code>/action", methods=["POST"])
@login_required
def action(code):
    alert = Alert.query.filter_by(alert_code=code).first_or_404()
    wants_json = request.is_json or request.accept_mimetypes.best == "application/json"
    data = request.get_json(silent=True) if request.is_json else request.form
    try:
        alerts_service.apply_action(alert, current_user, data.get("action", ""), data.get("comment", ""),
                                    data.get("escalate_to", ""))
        db.session.commit()
    except PermissionError as exc:
        db.session.rollback()
        if wants_json:
            return jsonify(ok=False, error=str(exc)), 403
        abort(403)
    except ValueError as exc:
        db.session.rollback()
        if wants_json:
            return jsonify(ok=False, error=str(exc)), 400
        flash(str(exc), "error")
        return redirect(request.referrer or url_for("alerts.list_alerts"))
    if wants_json:
        return jsonify(ok=True, status=alert.status)
    flash(f"{alert.alert_code}: {alert.status.lower()}.", "success")
    return redirect(request.referrer or url_for("alerts.list_alerts"))


@bp.route("/api/alerts/feed")
@login_required
def feed():
    since = request.args.get("since", 0, type=int)
    items = alerts_service.feed(current_user, since)
    active = alerts_service.visible_alerts(current_user).filter(Alert.status.in_(("Active", "Escalated"))).count()
    latest = alerts_service.visible_alerts(current_user).with_entities(db.func.max(Alert.id)).scalar() or 0
    return jsonify(
        latest_id=int(latest),
        active=active,
        alerts=[{
            "id": a.id, "code": a.alert_code, "category": a.category, "severity": a.severity,
            "message": a.message, "created_at": iso(a.created_at), "occurrences": a.occurrences,
            "url": url_for("analysis.record_detail", audio_id=a.record.audio_id),
            "can_act": can_act_on_alert(current_user, a),
        } for a in reversed(items)],
    )
