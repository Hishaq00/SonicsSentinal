"""Serves the exported Google Teachable Machine model files to the browser,
receives the browser's model-load status and serves the GTM training samples."""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_from_directory
from flask_login import current_user, login_required

from config import GTM_MODEL_DIR, GTM_SAMPLES_DIR

from ..extensions import db
from ..models import ModelVersion
from ..security import permission_required
from ..services import audit, gtm_service, monitoring

bp = Blueprint("gtm", __name__)
ALLOWED = {".json", ".bin"}


@bp.route("/gtm/model/<path:filename>")
@login_required
def model_file(filename):
    if "/" in filename or "\\" in filename or not any(filename.endswith(ext) for ext in ALLOWED):
        abort(404)
    if not (GTM_MODEL_DIR / filename).exists():
        abort(404)
    mimetype = "application/json" if filename.endswith(".json") else "application/octet-stream"
    resp = send_from_directory(GTM_MODEL_DIR, filename, mimetype=mimetype, max_age=0)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@bp.route("/api/gtm/status", methods=["POST"])
@login_required
def report_status():
    """The browser reports whether the GTM model loaded and which labels/version it has."""
    data = request.get_json(silent=True) or {}
    if data.get("ok"):
        labels = [str(x) for x in (data.get("labels") or [])][:64]
        version = gtm_service.clean_version(data.get("version"))
        check = gtm_service.label_check(labels)
        known = ModelVersion.query.filter_by(model_type="gtm", version=version).first()
        if not known or not known.is_active:
            gtm_service.register_version(version, labels, str(data.get("source") or "")[:400],
                                         current_user.username, notes="" if check["ok"] else f"label mismatch: {check}")
            audit.log("MODEL_UPDATE", "gtm_model", version, {"action": "loaded in browser", "labels_ok": check["ok"]})
        if not check["ok"]:
            monitoring.on_model_failure("GTM", f"class names do not match: missing {check['missing']}, extra {check['extra']}")
        db.session.commit()
        return jsonify(ok=True, labels_ok=check["ok"], check=check)
    monitoring.on_model_failure("GTM", str(data.get("error") or "model failed to load")[:300])
    db.session.commit()
    return jsonify(ok=True)


@bp.route("/gtm/samples/<path:filename>")
@permission_required("admin")
def training_sample(filename):
    target = (GTM_SAMPLES_DIR / filename).resolve()
    if GTM_SAMPLES_DIR.resolve() not in target.parents or not target.exists() or target.suffix.lower() not in (".wav", ".json"):
        abort(404)
    return send_from_directory(target.parent, target.name, max_age=0)
