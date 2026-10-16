"""Role-based access, role dashboards, manual review / override, alert rules,
batch permission and data export (route level, no trained model needed)."""
from __future__ import annotations

import io
import json

import pytest

from .conftest import login

ROLES = ["user", "reviewer", "security", "maintenance", "admin"]


def _make_record(app, audio_id="AUD-T-0001", category="Gunshot", severity="Critical", needs_review=True,
                 python_top="Gunshot", gtm_top="Vehicle Horn", quality="Poor", alert=True, owner="user"):
    """Insert a classified record (and optionally an alert) the way the pipeline stores it."""
    from alert_rules import rules_by_category
    from src.extensions import db
    from src.models import Alert, AudioRecord, User, utcnow

    with app.app_context():
        u = User.query.filter_by(username=owner).first()
        rec = AudioRecord(audio_id=audio_id, user_id=u.id, source="upload", original_filename=f"{audio_id}.wav",
                          status="Manual Review" if needs_review else "Alert Generated", gtm_status="completed",
                          python_top=python_top, python_confidence=0.71, gtm_top=gtm_top, gtm_confidence=0.64,
                          class_match=python_top == gtm_top, final_category=category, final_confidence=0.68,
                          consistency="Model Disagreement" if python_top != gtm_top else "Acceptable Match",
                          severity=severity, quality_label=quality, needs_review=needs_review,
                          review_reasons=["Different predictions from the two models", "Poor audio quality"]
                          if needs_review else [], classified_at=utcnow())
        db.session.add(rec)
        db.session.flush()
        if alert:
            rule = rules_by_category()[category]
            db.session.add(Alert(alert_code=f"ALR-{audio_id}", record_id=rec.id, category=category,
                                 severity=severity, source="upload", confidence=0.68, message=f"{category} test",
                                 recommended_action=rule["recommended_action"],
                                 responsible_roles=rule["responsible_roles"]))
        db.session.commit()
        return rec.id


# --------------------------------------------------------------------------- #
# Role-based access control and per-role dashboards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("role,marker", [
    ("user", b">Overview</h1>"),
    ("reviewer", b">Review desk</h1>"),
    ("security", b">Security console</h1>"),
    ("maintenance", b">Maintenance console</h1>"),
])
def test_each_role_lands_on_its_own_dashboard(app, client, role, marker):
    _make_record(app)
    login(client, role)
    r = client.get("/dashboard")
    assert r.status_code == 200 and marker in r.data


def test_admin_dashboard_is_system_overview(client):
    login(client, "admin")
    r = client.get("/dashboard")
    assert r.status_code == 302 and r.headers["Location"].endswith("/admin/")
    assert b"System overview" in client.get("/admin/").data


def test_operators_can_still_open_the_site_overview(client):
    login(client, "security")
    assert b">Overview</h1>" in client.get("/overview").data


def test_security_console_shows_only_security_categories(app, client):
    _make_record(app, "AUD-T-0001", "Gunshot", "Critical", needs_review=False)
    _make_record(app, "AUD-T-0002", "Machinery Fault", "High", needs_review=False, python_top="Machinery Fault",
                 gtm_top="Machinery Fault", quality="Good")
    login(client, "security")
    page = client.get("/dashboard").data
    assert b"ALR-AUD-T-0001" in page and b"ALR-AUD-T-0002" not in page
    client.post("/logout")
    login(client, "maintenance")
    page = client.get("/dashboard").data
    assert b"ALR-AUD-T-0002" in page and b"ALR-AUD-T-0001" not in page


def test_operator_can_acknowledge_from_console(app, client):
    _make_record(app, needs_review=False)
    login(client, "security")
    r = client.post("/alerts/ALR-AUD-T-0001/action", data={"action": "acknowledge"},
                    headers={"Referer": "/dashboard"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")
    from src.models import Alert
    with app.app_context():
        assert Alert.query.filter_by(alert_code="ALR-AUD-T-0001").one().status == "Acknowledged"


def test_maintenance_cannot_act_on_security_alert(app, client):
    _make_record(app, needs_review=False)
    login(client, "maintenance")
    r = client.post("/alerts/ALR-AUD-T-0001/action", data={"action": "acknowledge"})
    assert r.status_code == 403


@pytest.mark.parametrize("path,allowed", [
    ("/review", {"reviewer", "admin"}),
    ("/analytics", {"reviewer", "security", "maintenance", "admin"}),
    ("/admin/", {"admin"}),
    ("/admin/rules", {"admin"}),
    ("/admin/settings", {"admin"}),
    ("/history/export.csv", {"admin"}),
])
def test_permission_matrix(client, path, allowed):
    for role in ROLES:
        login(client, role)
        code = client.get(path).status_code
        assert (code == 200) == (role in allowed), f"{role} -> {path}: {code}"
        client.post("/logout")


def test_nav_labels_follow_role(client):
    login(client, "maintenance")
    page = client.get("/overview").data
    assert b"Maintenance console" in page and b"Site overview" in page and b"Users &amp; roles" not in page


# --------------------------------------------------------------------------- #
# Manual review workflow
# --------------------------------------------------------------------------- #
def test_disagreement_goes_to_review_queue_and_desk(app, client):
    _make_record(app)
    login(client, "reviewer")
    assert b"AUD-T-0001" in client.get("/review").data
    desk = client.get("/dashboard").data
    assert b"AUD-T-0001" in desk and b"Different predictions from the two models" in desk


def test_reviewer_override_keeps_model_outputs(app, client):
    _make_record(app)
    login(client, "reviewer")
    r = client.post("/records/AUD-T-0001/review", data={
        "decision": "corrected", "category": "Vehicle Horn", "comment": "Car backfire, not a gun.",
        "recommended_action": "No action."})
    assert r.status_code == 302
    from src.models import AudioRecord, AuditLog, Review
    with app.app_context():
        rec = AudioRecord.query.filter_by(audio_id="AUD-T-0001").one()
        assert rec.reviewed_category == "Vehicle Horn" and not rec.needs_review and rec.status == "Reviewed"
        assert rec.final_category == "Gunshot" and rec.python_top == "Gunshot" and rec.gtm_top == "Vehicle Horn"
        rv = Review.query.one()
        assert rv.is_override and rv.comment == "Car backfire, not a gun."
        assert AuditLog.query.filter_by(action="OVERRIDE").count() == 1
        assert all(a.status == "Dismissed" for a in rec.alerts)  # alert for the rejected category closed
    assert b"0 open" in client.get("/review").data  # left the queue (still listed under "Recently reviewed")


def test_normal_user_cannot_review(app, client):
    _make_record(app)
    login(client, "user")
    assert client.post("/records/AUD-T-0001/review", data={"decision": "confirmed"}).status_code == 403


def test_dismissed_alert_is_sent_to_review(app, client):
    _make_record(app, needs_review=False)
    login(client, "security")
    client.post("/alerts/ALR-AUD-T-0001/action", data={"action": "dismiss", "comment": "door slam"})
    from src.models import AudioRecord
    with app.app_context():
        rec = AudioRecord.query.filter_by(audio_id="AUD-T-0001").one()
        assert rec.needs_review and any("false alarm" in r for r in rec.review_reasons)


# --------------------------------------------------------------------------- #
# Configurable alert rules
# --------------------------------------------------------------------------- #
def _rules_form(doc, **changes):
    form = {"csrf_token": ""}
    for i, r in enumerate(doc["rules"]):
        p = f"r{i}."
        r = {**r, **changes.get(r["category"], {})}
        form[p + "category"] = r["category"]
        for k in ("severity", "min_audio_quality", "manual_review_condition", "recommended_action"):
            form[p + k] = r[k]
        for k in ("min_confidence", "min_top_two_margin", "required_consecutive_detections", "repeat_window_seconds"):
            form[p + k] = str(r[k])
        for k in ("critical", "alert_enabled", "require_model_agreement"):
            if r.get(k):
                form[p + k] = "1"
        form[p + "responsible_roles"] = r["responsible_roles"]
        esc = r["escalation"]
        form.update({p + "esc_min_repeats": str(esc.get("min_repeats", 3)), p + "esc_within_seconds": str(esc.get("within_seconds", 30)),
                     p + "esc_min_confidence": str(esc.get("min_confidence", 0.8)), p + "esc_severity": esc.get("severity", "High")})
    return form


def test_admin_changes_threshold_and_consecutive_rule(app, client, tmp_path, monkeypatch):
    import alert_rules.rules as rules_mod
    from alert_rules import load_rules
    from config import RULES_FILE as REAL_RULES_FILE

    real = REAL_RULES_FILE.read_bytes()
    rules_file = tmp_path / "alert_rules.json"
    rules_file.write_bytes(real)
    monkeypatch.setattr(rules_mod, "RULES_FILE", rules_file)
    import src.blueprints.admin as admin_bp
    monkeypatch.setattr(admin_bp, "RULES_DIR", tmp_path)

    doc = load_rules()
    login(client, "admin")
    form = _rules_form(doc, **{"Gunshot": {"min_confidence": 0.9, "required_consecutive_detections": 3,
                                           "severity": "High"}})
    r = client.post("/admin/rules", data=form)
    assert r.status_code == 302, r.data[:500]
    saved = {x["category"]: x for x in json.loads(rules_file.read_text(encoding="utf-8"))["rules"]}
    assert saved["Gunshot"]["min_confidence"] == 0.9
    assert saved["Gunshot"]["required_consecutive_detections"] == 3
    assert saved["Gunshot"]["severity"] == "High"
    assert (tmp_path / "history").exists()
    assert REAL_RULES_FILE.read_bytes() == real  # the project's rules are untouched


def test_all_five_severities_available(client):
    from config import SEVERITY_LEVELS
    assert set(SEVERITY_LEVELS) == {"Informational", "Low", "Medium", "High", "Critical"}


# --------------------------------------------------------------------------- #
# Batch processing and data export
# --------------------------------------------------------------------------- #
def test_batch_upload_requires_operator_role(client):
    login(client, "user")
    assert client.post("/api/batches", json={"files": 3}).status_code == 403
    client.post("/logout")
    login(client, "security")
    r = client.post("/api/batches", json={"files": 3})
    assert r.status_code == 201 and r.get_json()["batch"].startswith("BAT")
    assert client.post("/api/batches", json={"files": 0}).status_code == 400


def test_admin_exports_csv_and_excel(app, client):
    _make_record(app)
    login(client, "admin")
    csv_r = client.get("/history/export.csv")
    assert csv_r.status_code == 200 and b"AUD-T-0001" in csv_r.data
    xlsx = client.get("/history/export.xlsx")
    assert xlsx.status_code == 200 and xlsx.data[:2] == b"PK"  # zip container = .xlsx
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(xlsx.data))
    assert any("AUD-T-0001" in str(c.value) for row in wb.active.iter_rows() for c in row)
    assert client.get("/history/export.pdf").status_code == 400


def test_admin_dashboard_trend_data(app, client):
    _make_record(app)
    login(client, "admin")
    d = client.get("/api/analytics?days=14").get_json()
    assert sum(sum(v) for v in d["trend"]["series"].values()) >= 1
