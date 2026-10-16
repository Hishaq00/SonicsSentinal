"""Administrator area: dashboard, users, settings, alert rules, models, dataset,
audit trail, system (anomaly) alerts, data retention and model-comparison runs."""
from __future__ import annotations

import io
import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path

from flask import (Blueprint, Response, abort, current_app, flash, jsonify, redirect, render_template, request,
                   send_file, url_for)
from flask_login import current_user
from sqlalchemy import func

from alert_rules.rules import MANUAL_REVIEW_OPTIONS, ROLE_KEYS, load_rules, save_rules, validate_rules
from config import (DATA_DIR, METADATA_CSV, QUALITY_LEVELS, REPORTS_DIR, RULES_DIR, SEVERITY_LEVELS, class_info,
                    class_names)
from python_models.inference import ModelNotAvailable, get_classifier

from ..extensions import db
from ..models import (ROLES, Alert, AudioRecord, AuditLog, EvaluationRun, MicSession, ModelVersion, SystemAlert,
                      User, utcnow)
from ..security import permission_required, role_matrix
from ..services import analytics, audit, evaluation_service, gtm_service, pipeline, reports, retention, storage
from ..services.audit import ACTIONS
from ..services.settings_service import get_settings, schema_by_section, update_settings

bp = Blueprint("admin", __name__, url_prefix="/admin")


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
@bp.route("/")
@permission_required("admin")
def dashboard():
    clf = get_classifier()
    storage_bytes = sum(p.stat().st_size for p in storage.root().rglob("*") if p.is_file())
    db_path = current_app.config["SQLALCHEMY_DATABASE_URI"].replace("sqlite:///", "")
    db_size = Path(db_path).stat().st_size if db_path and Path(db_path).exists() else None
    pending_roles = [u for u in User.query.order_by(User.created_at.desc()).all()
                     if (u.preferences or {}).get("requested_role") and u.role != (u.preferences or {}).get("requested_role")]
    return render_template(
        "admin/dashboard.html",
        kpis=analytics.kpis(current_user, 30),
        categories=analytics.category_counts(current_user, 30),
        quality=analytics.quality_distribution(current_user, 30),
        pairs=analytics.disagreement_pairs(current_user, 30),
        hourly=analytics.hourly_activity(current_user, 24),
        traces=analytics.recent_traces(current_user, 6),
        consistency=analytics.consistency_counts(current_user, 30),
        confidence=analytics.confidence_distribution(current_user, 30),
        class_colors={name: info.get("color", "#8a8578") for name, info in class_info().items()},
        system_alerts=SystemAlert.query.filter_by(resolved=False).order_by(SystemAlert.last_seen_at.desc()).limit(8).all(),
        python_ready=clf.available,
        python_card=clf.card if clf.available else None,
        gtm=gtm_service.client_config(clf.classes if clf.available else None),
        gtm_versions=ModelVersion.query.filter_by(model_type="gtm").order_by(ModelVersion.registered_at.desc()).limit(3).all(),
        active_sessions=MicSession.query.filter(MicSession.ended_at.is_(None)).count(),
        users=User.query.count(),
        storage_bytes=storage_bytes,
        db_size=db_size,
        pending_gtm=AudioRecord.query.filter_by(gtm_status="pending").count(),
        retention_last=retention.last_run(),
        pending_roles=pending_roles,
        recent_audit=AuditLog.query.order_by(AuditLog.created_at.desc()).limit(8).all(),
    )


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
@bp.route("/users")
@permission_required("admin")
def users():
    q = (request.args.get("q") or "").strip()
    query = User.query
    if q:
        like = f"%{q}%"
        query = query.filter((User.username.ilike(like)) | (User.full_name.ilike(like)) | (User.email.ilike(like))
                             | (User.user_code.ilike(like)))
    return render_template("admin/users.html", users=query.order_by(User.created_at.desc()).all(), roles=ROLES,
                           matrix=role_matrix(), q=q, now=utcnow())


@bp.route("/users/<code>", methods=["POST"])
@permission_required("admin")
def update_user(code):
    user = User.query.filter_by(user_code=code).first_or_404()
    action = request.form.get("action")
    temp_password = None
    if action == "role":
        role = request.form.get("role")
        if role not in ROLES:
            abort(400)
        if user.id == current_user.id and role != "admin":
            flash("You cannot remove your own administrator role.", "error")
            return redirect(url_for("admin.users"))
        before = user.role
        user.role = role
        prefs = dict(user.preferences or {})
        prefs.pop("requested_role", None)
        user.preferences = prefs
        audit.log("USER_UPDATE", "user", user.user_code, {"role": {"from": before, "to": role}})
    elif action == "toggle":
        if user.id == current_user.id:
            flash("You cannot disable your own account.", "error")
            return redirect(url_for("admin.users"))
        user.active_account = not user.active_account
        audit.log("USER_UPDATE", "user", user.user_code, {"active": user.active_account})
    elif action == "unlock":
        user.locked_until = None
        user.failed_logins = 0
        audit.log("USER_UPDATE", "user", user.user_code, {"unlocked": True})
    elif action == "reset_password":
        temp_password = "Ss-" + secrets.token_urlsafe(8) + "1"
        user.set_password(temp_password)
        audit.log("USER_UPDATE", "user", user.user_code, {"password_reset": True})
    elif action == "reject_role":
        prefs = dict(user.preferences or {})
        prefs.pop("requested_role", None)
        user.preferences = prefs
        audit.log("USER_UPDATE", "user", user.user_code, {"role_request": "rejected"})
    else:
        abort(400)
    db.session.commit()
    if temp_password:
        flash(f"Temporary password for {user.username}: {temp_password} (shown once; ask the user to change it).", "success")
    else:
        flash(f"{user.username} updated.", "success")
    return redirect(request.referrer or url_for("admin.users"))


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@bp.route("/settings", methods=["GET", "POST"])
@permission_required("admin")
def settings():
    if request.method == "POST":
        changes: dict = {}
        schema = schema_by_section()
        for section, fields in schema.items():
            for f in fields:
                name = f"{section}.{f['key']}"
                if f["type"] == "bool":
                    changes.setdefault(section, {})[f["key"]] = bool(request.form.get(name))
                elif name in request.form:
                    changes.setdefault(section, {})[f["key"]] = request.form.get(name)
        if "validation.allowed_formats" in request.form or request.form.getlist("formats"):
            changes.setdefault("validation", {})["allowed_formats"] = request.form.getlist("formats")
        diff, errors = update_settings(changes, current_user.username)
        if errors:
            db.session.rollback()
            for e in errors:
                flash(e, "error")
        else:
            if diff:
                audit.log("SETTINGS_CHANGE", "settings", None, diff)
            db.session.commit()
            flash("Settings saved." if diff else "No changes.", "success")
        return redirect(url_for("admin.settings"))
    from audio_preprocessing.loader import SUPPORTED_EXTENSIONS

    return render_template("admin/settings.html", settings=get_settings(), schema=schema_by_section(),
                           supported=SUPPORTED_EXTENSIONS, retention_last=retention.last_run())


@bp.route("/retention/run", methods=["POST"])
@permission_required("admin")
def run_retention():
    summary = retention.run_retention(current_user.username)
    flash(f"Retention clean-up done: {summary['records_deleted']} records, {summary['audio_deleted']} uploaded "
          f"audio files and {summary['live_audio_deleted']} live audio files removed.", "success")
    return redirect(request.referrer or url_for("admin.settings"))


# --------------------------------------------------------------------------- #
# Alert rules
# --------------------------------------------------------------------------- #
@bp.route("/rules", methods=["GET", "POST"])
@permission_required("admin")
def rules():
    doc = load_rules()
    if request.method == "POST":
        new_rules = []
        for i, old in enumerate(doc["rules"]):
            p = f"r{i}."
            if request.form.get(p + "category") != old["category"]:
                abort(400)
            rule = dict(old)
            rule["critical"] = bool(request.form.get(p + "critical"))
            rule["alert_enabled"] = bool(request.form.get(p + "alert_enabled"))
            rule["severity"] = request.form.get(p + "severity", old["severity"])
            rule["min_confidence"] = _num(request.form.get(p + "min_confidence"), old["min_confidence"])
            rule["min_top_two_margin"] = _num(request.form.get(p + "min_top_two_margin"), old["min_top_two_margin"])
            rule["required_consecutive_detections"] = int(_num(request.form.get(p + "required_consecutive_detections"),
                                                               old["required_consecutive_detections"]))
            rule["repeat_window_seconds"] = int(_num(request.form.get(p + "repeat_window_seconds"), old["repeat_window_seconds"]))
            rule["require_model_agreement"] = bool(request.form.get(p + "require_model_agreement"))
            rule["min_audio_quality"] = request.form.get(p + "min_audio_quality", old["min_audio_quality"])
            rule["manual_review_condition"] = request.form.get(p + "manual_review_condition", old["manual_review_condition"])
            rule["recommended_action"] = (request.form.get(p + "recommended_action") or "").strip()[:600]
            rule["responsible_roles"] = [r for r in request.form.getlist(p + "responsible_roles") if r in ROLE_KEYS]
            esc = dict(old.get("escalation", {}))
            esc["enabled"] = bool(request.form.get(p + "esc_enabled"))
            esc["min_repeats"] = int(_num(request.form.get(p + "esc_min_repeats"), esc.get("min_repeats", 3)))
            esc["within_seconds"] = int(_num(request.form.get(p + "esc_within_seconds"), esc.get("within_seconds", 30)))
            esc["min_confidence"] = _num(request.form.get(p + "esc_min_confidence"), esc.get("min_confidence", 0.8))
            esc["severity"] = request.form.get(p + "esc_severity", esc.get("severity", "High"))
            esc["generate_alert"] = bool(request.form.get(p + "esc_generate_alert"))
            esc["notify"] = (request.form.get(p + "esc_notify") or "").strip()[:160]
            rule["escalation"] = esc
            new_rules.append(rule)
        candidate = {**doc, "rules": new_rules, "notes": doc.get("notes", "")}
        errors = validate_rules(candidate)
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("admin/rules.html", doc=candidate, **_rule_choices()), 400
        before = {r["category"]: r for r in doc["rules"]}
        changed = {r["category"]: {k: {"from": before[r["category"]].get(k), "to": v} for k, v in r.items()
                                   if before[r["category"]].get(k) != v}
                   for r in new_rules if r != before.get(r["category"])}
        saved = save_rules(candidate, current_user.username)
        audit.log("RULES_CHANGE", "alert_rules", f"v{saved['version']}", {"changes": changed})
        db.session.commit()
        flash(f"Alert rules saved as version {saved['version']} (previous version kept in alert_rules/history/).", "success")
        return redirect(url_for("admin.rules"))
    return render_template("admin/rules.html", doc=doc, **_rule_choices())


def _num(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rule_choices() -> dict:
    history = sorted((RULES_DIR / "history").glob("alert_rules_v*.json"), reverse=True) if (RULES_DIR / "history").exists() else []
    return {"severities": SEVERITY_LEVELS, "qualities": QUALITY_LEVELS, "review_options": MANUAL_REVIEW_OPTIONS,
            "role_keys": ROLE_KEYS, "roles": ROLES, "history": [h.name for h in history[:10]], "class_info": class_info()}


@bp.route("/rules/download")
@permission_required("admin")
def download_rules():
    return send_file(RULES_DIR / "alert_rules.json", mimetype="application/json", as_attachment=True,
                     download_name="alert_rules.json")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@bp.route("/models")
@permission_required("admin")
def models():
    clf = get_classifier()
    cards = clf.list_models()
    active = clf.active_pointer()
    meta = gtm_service.read_local_metadata()
    labels = gtm_service.labels_from_metadata(meta)
    expected = clf.classes if clf.available else class_names()
    report_dir = None
    if clf.available:
        report_dir = clf.card.get("report_dir")
    images = []
    if report_dir:
        d = REPORTS_DIR.parent / report_dir
        images = sorted(p.name for p in d.glob("confusion_matrix_*.png")) if d.exists() else []
    return render_template(
        "admin/models.html", cards=cards, active=active, card=clf.card if clf.available else None,
        images=images, report_dir=report_dir, gtm=gtm_service.client_config(expected),
        gtm_settings=get_settings()["gtm"], gtm_files=gtm_service.local_files(), gtm_meta=meta, gtm_labels=labels,
        label_check=gtm_service.label_check(labels, expected) if labels else None,
        gtm_versions=ModelVersion.query.filter_by(model_type="gtm").order_by(ModelVersion.registered_at.desc()).all(),
        py_versions=ModelVersion.query.filter_by(model_type="python").order_by(ModelVersion.registered_at.desc()).all(),
        feature_groups=_feature_groups(clf),
    )


def _feature_groups(clf):
    from feature_extraction.features import FeatureConfig, feature_groups

    try:
        return feature_groups(clf.feature_config if clf.available else FeatureConfig())
    except ModelNotAvailable:
        return {}


@bp.route("/models/activate", methods=["POST"])
@permission_required("admin")
def activate_model():
    version = request.form.get("version", "")
    clf = get_classifier()
    try:
        clf.activate(version)
    except ModelNotAvailable as exc:
        flash(str(exc), "error")
        return redirect(url_for("admin.models"))
    sync_python_versions(current_user.username)
    audit.log("MODEL_UPDATE", "python_model", version, {"action": "activated"})
    db.session.commit()
    flash(f"Python model {version} is now active.", "success")
    return redirect(url_for("admin.models"))


@bp.route("/models/gtm", methods=["POST"])
@permission_required("admin")
def configure_gtm():
    changes = {"gtm": {"enabled": bool(request.form.get("enabled")), "source": request.form.get("source", "local"),
                       "url": request.form.get("url", ""), "version_label": request.form.get("version_label", "")}}
    diff, errors = update_settings(changes, current_user.username)
    if errors:
        db.session.rollback()
        for e in errors:
            flash(e, "error")
        return redirect(url_for("admin.models"))
    if diff:
        audit.log("MODEL_UPDATE", "gtm_model", changes["gtm"].get("version_label") or "gtm", {"settings": diff})
    db.session.commit()
    flash("GTM configuration saved. The browser loads the model on the next analysis.", "success")
    return redirect(url_for("admin.models"))


@bp.route("/reports-file/<path:filename>")
@permission_required("admin")
def report_file(filename):
    path = (REPORTS_DIR / filename).resolve()
    if REPORTS_DIR.resolve() not in path.parents or not path.exists():
        abort(404)
    return send_file(path)


def sync_python_versions(username: str = "system") -> None:
    """Mirror the model cards in python_models/saved/ into the model_versions table."""
    clf = get_classifier()
    active = (clf.active_pointer() or {}).get("version")
    for card in clf.list_models():
        mv = ModelVersion.query.filter_by(model_type="python", version=card["version"]).first()
        if mv is None:
            mv = ModelVersion(model_type="python", version=card["version"], algorithm=card.get("algorithm_label"),
                              source=f"python_models/saved/{card.get('file')}", labels=card.get("classes"),
                              metrics={"test": (card.get("metrics", {}).get("test") or {}).get("recording", {}).get("accuracy"),
                                       "f1_macro": (card.get("metrics", {}).get("test") or {}).get("recording", {}).get("f1_macro")},
                              registered_by=username)
            db.session.add(mv)
            audit.log("MODEL_UPDATE", "python_model", card["version"], {"action": "registered"}, username=username)
        mv.is_active = card["version"] == active
    db.session.flush()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@bp.route("/dataset")
@permission_required("admin")
def dataset():
    stats_path = DATA_DIR / "dataset_statistics.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    sample = []
    if METADATA_CSV.exists():
        import pandas as pd

        df = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
        sample = df[df["status"] == "original"].groupby("class_label").head(2).head(20).to_dict("records")
    feat_cfg = DATA_DIR / "features" / "feature_config.json"
    return render_template("admin/dataset.html", stats=stats, sample=sample, class_info=class_info(),
                           features=json.loads(feat_cfg.read_text()) if feat_cfg.exists() else None)


@bp.route("/dataset/metadata.csv")
@permission_required("admin")
def dataset_metadata():
    if not METADATA_CSV.exists():
        abort(404)
    audit.log("EXPORT", "dataset_metadata", None, {"format": "csv"})
    db.session.commit()
    return send_file(METADATA_CSV, mimetype="text/csv", as_attachment=True, download_name="dataset_metadata.csv")


# --------------------------------------------------------------------------- #
# Audit trail and system alerts
# --------------------------------------------------------------------------- #
@bp.route("/audit")
@permission_required("admin")
def audit_trail():
    action = request.args.get("action") or ""
    username = (request.args.get("user") or "").strip()
    q = AuditLog.query
    if action:
        q = q.filter(AuditLog.action == action)
    if username:
        q = q.filter(AuditLog.username.ilike(f"%{username}%"))
    for key, op in (("date_from", ">="), ("date_to", "<")):
        try:
            d = datetime.strptime(request.args.get(key, ""), "%Y-%m-%d")
            q = q.filter(AuditLog.created_at >= d if op == ">=" else AuditLog.created_at < d + timedelta(days=1))
        except ValueError:
            pass
    if request.args.get("format") == "csv":
        rows = [[a.created_at.strftime("%Y-%m-%d %H:%M:%S"), a.username, a.action, a.target_type, a.target_id,
                 a.ip_address, json.dumps(a.details or {})] for a in q.order_by(AuditLog.created_at.desc()).limit(50000)]
        audit.log("EXPORT", "audit_trail", None, {"rows": len(rows)})
        db.session.commit()
        return Response(reports.to_csv(["time_utc", "user", "action", "target_type", "target_id", "ip", "details"], rows),
                        mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=audit_trail.csv"})
    page = max(1, request.args.get("page", 1, type=int))
    pagination = q.order_by(AuditLog.created_at.desc()).paginate(page=page, per_page=50, error_out=False)
    counts = dict(db.session.query(AuditLog.action, func.count()).group_by(AuditLog.action).all())
    return render_template("admin/audit.html", logs=pagination.items, pagination=pagination, actions=ACTIONS,
                           counts=counts, action=action, username=username)


@bp.route("/system-alerts")
@permission_required("admin")
def system_alerts():
    show = request.args.get("show", "open")
    q = SystemAlert.query
    if show == "open":
        q = q.filter_by(resolved=False)
    return render_template("admin/system_alerts.html", alerts=q.order_by(SystemAlert.last_seen_at.desc()).limit(200).all(),
                           show=show)


@bp.route("/system-alerts/<int:alert_id>/resolve", methods=["POST"])
@permission_required("admin")
def resolve_system_alert(alert_id):
    sa = db.session.get(SystemAlert, alert_id) or abort(404)
    sa.resolved = True
    sa.resolved_by = current_user.username
    sa.resolved_at = utcnow()
    audit.log("SYSTEM_ALERT_RESOLVED", "system_alert", sa.id, {"kind": sa.kind})
    db.session.commit()
    flash("System alert resolved.", "success")
    return redirect(request.referrer or url_for("admin.system_alerts"))


# --------------------------------------------------------------------------- #
# Model prediction & confidence comparison runs
# --------------------------------------------------------------------------- #
@bp.route("/evaluation", methods=["GET", "POST"])
@permission_required("evaluation")
def evaluation():
    if request.method == "POST":
        try:
            run = evaluation_service.create_run(current_user, request.form.get("per_class", 10, type=int))
        except (ValueError, ModelNotAvailable, FileNotFoundError) as exc:
            flash(str(exc), "error")
            return redirect(url_for("admin.evaluation"))
        return redirect(url_for("admin.evaluation_run", code=run.run_code))
    runs = EvaluationRun.query.order_by(EvaluationRun.created_at.desc()).all()
    pool = 0
    try:
        pool = len(evaluation_service.test_pool())
    except FileNotFoundError:
        pass
    return render_template("admin/evaluation_list.html", runs=runs, pool=pool)


@bp.route("/evaluation/<code>")
@permission_required("evaluation")
def evaluation_run(code):
    run = EvaluationRun.query.filter_by(run_code=code).first_or_404()
    clf = get_classifier()
    summary = (run.summary or {}).get("result") or (evaluation_service.compute_summary(run) if run.completed else None)
    head, rows = evaluation_service.report_rows(run) if run.completed else ([], [])
    return render_template("admin/evaluation_run.html", run=run, summary=summary, head=head, rows=rows,
                           gtm=gtm_service.client_config(clf.classes if clf.available else None),
                           classes=class_names())


@bp.route("/api/evaluation/<code>/next", methods=["POST"])
@permission_required("evaluation")
def evaluation_next(code):
    run = EvaluationRun.query.filter_by(run_code=code).first_or_404()
    if run.status == "completed":
        return jsonify(done=True)
    pipeline.finalize_stale(minutes=5)
    return jsonify(evaluation_service.process_next(run, current_user))


@bp.route("/api/evaluation/<code>/finish", methods=["POST"])
@permission_required("evaluation")
def evaluation_finish(code):
    run = EvaluationRun.query.filter_by(run_code=code).first_or_404()
    evaluation_service.finish(run)
    return jsonify(ok=True, url=url_for("admin.evaluation_run", code=code))


@bp.route("/evaluation/<code>/report.<fmt>")
@permission_required("evaluation")
def evaluation_report(code, fmt):
    run = EvaluationRun.query.filter_by(run_code=code).first_or_404()
    head, rows = evaluation_service.report_rows(run)
    summary = (run.summary or {}).get("result") or evaluation_service.compute_summary(run)
    audit.log("EXPORT", "evaluation", run.run_code, {"format": fmt, "rows": len(rows)})
    db.session.commit()
    name = f"model_comparison_{run.run_code}"
    if fmt == "csv":
        return Response(reports.to_csv(head, rows), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={name}.csv"})
    if fmt == "xlsx":
        labels = summary.get("labels", class_names())
        per_class = [[p["class"], p["n"], p["python_accuracy"], p["gtm_accuracy"], p["final_accuracy"], p["agreement"],
                      p["mean_confidence_difference"]] for p in summary.get("per_class", [])]
        overview = [["Recordings", summary["n"]], ["With GTM result", summary["n_with_gtm"]],
                    ["Python accuracy", summary["python"]["accuracy"]], ["Python macro F1", summary["python"]["f1_macro"]],
                    ["GTM accuracy", summary["gtm"]["accuracy"]], ["GTM macro F1", summary["gtm"]["f1_macro"]],
                    ["Final accuracy", summary["final"]["accuracy"]], ["Agreement rate", summary["agreement_rate"]],
                    ["Mean confidence difference", summary["mean_confidence_difference"]],
                    ["Both correct", summary["both_correct"]], ["Only Python correct", summary["only_python_correct"]],
                    ["Only GTM correct", summary["only_gtm_correct"]], ["Both wrong", summary["both_wrong"]],
                    ["Python model", summary.get("python_version")], ["GTM model", summary.get("gtm_version")],
                    ["Summary", summary["text"]]]
        cm_rows = [[labels[i]] + row for i, row in enumerate(summary["python"]["confusion_matrix"])]
        gcm_rows = [[labels[i]] + row for i, row in enumerate(summary["gtm"]["confusion_matrix"])]
        data = reports.to_xlsx([
            ("Summary", ["Metric", "Value"], overview),
            ("Per class", ["Class", "Recordings", "Python accuracy", "GTM accuracy", "Final accuracy", "Agreement",
                           "Mean confidence difference"], per_class),
            ("Recordings", head, rows),
            ("Python confusion", ["Actual \\ Predicted"] + labels, cm_rows),
            ("GTM confusion", ["Actual \\ Predicted"] + labels, gcm_rows),
        ])
        return Response(data, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f"attachment; filename={name}.xlsx"})
    if fmt == "html":
        html = render_template("admin/evaluation_print.html", run=run, summary=summary, head=head, rows=rows,
                               classes=class_names())
        return Response(html, mimetype="text/html",
                        headers={"Content-Disposition": f"attachment; filename={name}.html"})
    abort(404)


# --------------------------------------------------------------------------- #
# GTM training helper
# --------------------------------------------------------------------------- #
@bp.route("/gtm-trainer")
@permission_required("admin")
def gtm_trainer():
    from config import GTM_SAMPLES_DIR

    manifest = GTM_SAMPLES_DIR / "manifest.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else None
    return render_template("admin/gtm_trainer.html", manifest=data, class_info=class_info())
