"""User dashboard and the chronological event timeline."""
from __future__ import annotations

import json
from collections import OrderedDict
from datetime import timedelta

from flask import Blueprint, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from alert_rules import rules_by_category
from config import DATA_DIR, MODELS_DIR, SEVERITY_LEVELS, class_info, class_names, load_classes

from ..models import Alert, AudioRecord, MicSession, utcnow
from ..security import can, can_act_on_alert
from ..services import analytics, gtm_service, role_dashboards
from ..services.alerts_service import visible_alerts
from .alerts import ESCALATION_TARGETS

bp = Blueprint("main", __name__)


@bp.route("/")
@bp.route("/landing")
@bp.route("/home")
def index():
    rules = rules_by_category()
    classes = [{**c, "severity": rules.get(c["name"], {}).get("severity", "Low"),
                "alerts": bool(rules.get(c["name"], {}).get("alert_enabled"))} for c in load_classes()["classes"]]
    return render_template("landing.html", classes=classes, facts=_landing_facts())


def _landing_facts() -> dict:
    """Real numbers for the public page: current model cards and the dataset statistics."""
    facts: dict = {"python": None, "gtm": None, "recordings": None}
    try:
        active = json.loads((MODELS_DIR / "active.json").read_text(encoding="utf-8"))
        card = json.loads((MODELS_DIR / f"{active['version']}.json").read_text(encoding="utf-8"))
        test = card["metrics"]["test"]["recording"]
        facts["python"] = {"version": card["version"], "accuracy": test["accuracy"], "f1": test["f1_macro"],
                           "features": card.get("n_features"), "test_n": test.get("n")}
    except (OSError, KeyError, ValueError):
        pass
    meta = gtm_service.read_local_metadata() or {}
    if meta.get("accuracy") is not None:
        facts["gtm"] = {"version": meta.get("modelName"), "accuracy": meta["accuracy"],
                        "f1": (meta.get("metrics") or {}).get("test_f1_macro")}
    try:
        stats = json.loads((DATA_DIR / "dataset_statistics.json").read_text(encoding="utf-8"))
        facts["recordings"] = sum(c.get("original", 0) + c.get("augmented", 0) for c in stats["classes"].values())
    except (OSError, KeyError, ValueError):
        pass
    return facts


ROLE_DASHBOARDS = {"reviewer": "reviewer", "security": "operator", "maintenance": "operator"}


@bp.route("/dashboard")
@login_required
def dashboard():
    """Each role lands on its own dashboard; normal users get the overview."""
    if current_user.role == "admin":
        return redirect(url_for("admin.dashboard"))
    kind = ROLE_DASHBOARDS.get(current_user.role)
    if kind == "reviewer":
        return render_template("dashboard/reviewer.html", d=role_dashboards.reviewer_dashboard(current_user))
    if kind == "operator":
        return render_template("dashboard/operator.html", d=role_dashboards.operator_dashboard(current_user),
                               targets=ESCALATION_TARGETS, can_act=can_act_on_alert)
    return overview()


@bp.route("/overview")
@login_required
def overview():
    """General overview: site-wide for operators, own recordings for normal users."""
    user = current_user
    scoped = analytics.scoped_records(user)
    recent_uploads = (analytics.scoped_records(user).filter(AudioRecord.user_id == user.id,
                                                            AudioRecord.source.in_(("upload", "batch")))
                      .order_by(AudioRecord.created_at.desc()).limit(8).all())
    current_detections = (scoped.filter(AudioRecord.source == "live", AudioRecord.final_category.isnot(None),
                                        AudioRecord.final_category != "Background Noise")
                          .order_by(AudioRecord.created_at.desc()).limit(8).all())
    critical = (visible_alerts(user).filter(Alert.severity.in_(("Critical", "High")))
                .order_by(Alert.created_at.desc()).limit(8).all())
    quality_warnings = (scoped.filter(AudioRecord.quality_label.in_(("Poor", "Unusable")))
                        .order_by(AudioRecord.created_at.desc()).limit(6).all())
    review_q = scoped.filter(AudioRecord.needs_review.is_(True))
    if not can(user, "review"):
        review_q = review_q.filter(AudioRecord.user_id == user.id)
    review_records = review_q.order_by(AudioRecord.created_at.desc()).limit(6).all()
    active_session = (MicSession.query.filter(MicSession.user_id == user.id, MicSession.ended_at.is_(None))
                      .order_by(MicSession.started_at.desc()).first())
    return render_template(
        "dashboard/user.html",
        kpis=analytics.kpis(user, 30),
        recent_uploads=recent_uploads,
        current_detections=current_detections,
        critical=critical,
        quality_warnings=quality_warnings,
        review_records=review_records,
        categories=analytics.category_counts(user, 30),
        timeline=analytics.critical_timeline(user, 24),
        hourly=analytics.hourly_activity(user, 24),
        traces=analytics.recent_traces(user, 5),
        class_colors={name: info.get("color", "#8a8578") for name, info in class_info().items()},
        active_session=active_session,
        now=utcnow(),
    )


@bp.route("/timeline")
@login_required
def timeline():
    days = min(90, max(1, request.args.get("days", 7, type=int)))
    category = request.args.get("category") or ""
    severity = request.args.get("severity") or ""
    since = utcnow() - timedelta(days=days)
    q = analytics.scoped_records(current_user).filter(
        AudioRecord.created_at >= since, AudioRecord.final_category.isnot(None),
        AudioRecord.final_category != "Background Noise")
    q = analytics.apply_filters(q, {"category": category, "severity": severity, "include_background": True})
    records = q.order_by(AudioRecord.created_at.desc()).limit(400).all()
    grouped: OrderedDict[str, list] = OrderedDict()
    for r in records:
        grouped.setdefault(r.created_at.strftime("%A %d %B %Y"), []).append(r)
    return render_template("dashboard/timeline.html", grouped=grouped, days=days, category=category,
                           severity=severity, classes=class_names(), severities=SEVERITY_LEVELS, total=len(records))
