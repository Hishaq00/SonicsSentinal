"""Security, database, duplicate / near-duplicate audio and live-microphone tests."""
from __future__ import annotations

import io
import time

import numpy as np
import pytest
import soundfile as sf

from audio_preprocessing.fingerprint import NEAR_DUPLICATE_BER, compare_fingerprints, compute_fingerprint

from .conftest import SR, impulse_in_noise, login, needs_model, noise, tone, upload, write


# --------------------------------------------------------------------------- #
# Security
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", ["/dashboard", "/history", "/alerts", "/review", "/admin/", "/live", "/analyze"])
def test_pages_require_login(client, path):
    r = client.get(path)
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_api_requires_login(client):
    assert client.post("/api/uploads").status_code in (302, 401)
    assert client.get("/api/alerts/feed").status_code in (302, 401)


def test_wrong_password_rejected_and_audited(app, client):
    r = login(client, "user", "not-the-password")
    assert r.status_code == 401
    from src.models import AuditLog
    with app.app_context():
        assert AuditLog.query.filter_by(action="LOGIN_FAILED").count() == 1


def test_account_locks_after_repeated_failures(app, client):
    for _ in range(5):
        login(client, "user", "wrong")
    r = login(client, "user")  # correct password, but locked now
    assert r.status_code == 429
    from src.models import SystemAlert
    with app.app_context():
        assert SystemAlert.query.filter(SystemAlert.kind.like("%login%")).count() >= 1


def test_passwords_are_hashed(app):
    from src.models import User
    with app.app_context():
        u = User.query.filter_by(username="admin").one()
        assert "Admin@2026" not in u.password_hash and u.check_password("Admin@2026")


def test_open_redirect_blocked(client):
    r = client.post("/login?next=https://evil.example/steal", data={"username": "user", "password": "User@2026"})
    assert r.status_code == 302 and "evil.example" not in r.headers["Location"]


def test_security_headers(client):
    r = client.get("/login")
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert "Content-Security-Policy" in r.headers


def test_csrf_enforced_when_enabled(tmp_path):
    from src import create_app
    from src.app_config import TestConfig

    class Cfg(TestConfig):
        SQLALCHEMY_DATABASE_URI = "sqlite:///" + (tmp_path / "csrf.db").as_posix()
        STORAGE_DIR = str(tmp_path / "storage")
        WTF_CSRF_ENABLED = True

    app = create_app(Cfg)
    r = app.test_client().post("/login", data={"username": "admin", "password": "Admin@2026"})
    assert r.status_code == 400  # missing CSRF token
    from src.extensions import db
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def test_user_cannot_open_someone_elses_record(app, client):
    from .test_roles_workflow import _make_record
    _make_record(app, owner="security")
    login(client, "user")
    assert client.get("/records/AUD-T-0001").status_code in (403, 404)


def test_normal_user_cannot_change_roles(client):
    login(client, "user")
    assert client.post("/admin/users/USR-0001", data={"action": "role", "role": "admin"}).status_code == 403


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
def test_record_alert_review_audit_are_linked(app):
    from src.extensions import db
    from src.models import Alert, AudioRecord, Review, User

    from .test_roles_workflow import _make_record
    rid = _make_record(app)
    with app.app_context():
        rec = db.session.get(AudioRecord, rid)
        rev = User.query.filter_by(username="reviewer").one()
        db.session.add(Review(record_id=rid, reviewer_id=rev.id, decision="confirmed",
                              original_category="Gunshot", reviewed_category="Gunshot"))
        db.session.commit()
        assert rec.alerts[0].record is rec and rec.reviews[0].reviewer.username == "reviewer"
        db.session.delete(rec)
        db.session.commit()
        assert Alert.query.count() == 0 and Review.query.count() == 0  # cascade


def test_audio_id_is_unique(app):
    from sqlalchemy.exc import IntegrityError

    from src.extensions import db

    from .test_roles_workflow import _make_record
    _make_record(app, alert=False)
    with pytest.raises(IntegrityError):
        _make_record(app, alert=False)
    with app.app_context():
        db.session.rollback()


def test_20000_records_stay_searchable(app, client):
    """SRS 1.7-2: at least 20,000 sound-event records without significant slow-down."""
    from src.extensions import db
    from src.models import AudioRecord, User, utcnow

    classes = ["Gunshot", "Vehicle Horn", "Background Noise", "Machinery Fault"]
    with app.app_context():
        uid = User.query.filter_by(username="admin").one().id
        now = utcnow()
        db.session.bulk_insert_mappings(AudioRecord, [
            {"audio_id": f"AUD-S-{i:06d}", "user_id": uid, "source": "upload", "original_filename": f"f{i}.wav",
             "final_category": classes[i % 4], "final_confidence": (i % 100) / 100, "severity": "Low",
             "quality_label": "Good", "status": "Classified", "gtm_status": "completed", "created_at": now}
            for i in range(20000)])
        db.session.commit()
        assert AudioRecord.query.count() == 20000
    login(client, "admin")
    client.get("/admin/")  # first request loads the model from disk (done at start-up in production)
    t = time.perf_counter()
    r = client.get("/history?category=Gunshot&conf_min=0.5")
    elapsed = time.perf_counter() - t
    assert r.status_code == 200 and elapsed < 5.0, f"history search took {elapsed:.2f} s"
    t = time.perf_counter()
    assert client.get("/admin/").status_code == 200
    assert time.perf_counter() - t < 5.0


# --------------------------------------------------------------------------- #
# Duplicate and near-duplicate audio
# --------------------------------------------------------------------------- #
def test_fingerprint_matches_volume_adjusted_and_trimmed_copy():
    y = impulse_in_noise(4.0, at=1.5, seed=3) + tone(4.0, 660, 0.1)
    fp, _ = compute_fingerprint(y, SR)
    quieter, _ = compute_fingerprint(y * 0.4, SR)
    trimmed, _ = compute_fingerprint(y[int(0.3 * SR):], SR)
    other, _ = compute_fingerprint(noise(4.0, 0.2, seed=9), SR)
    assert compare_fingerprints(fp, quieter)[0] <= NEAR_DUPLICATE_BER
    assert compare_fingerprints(fp, trimmed)[0] <= NEAR_DUPLICATE_BER
    assert compare_fingerprints(fp, other)[0] > NEAR_DUPLICATE_BER


@needs_model
def test_identical_upload_flagged_as_duplicate(app, client, no_gtm, tmp_path):
    path = write(tmp_path / "horn.wav", impulse_in_noise(3.0, seed=4) + tone(3.0, 420, 0.3))
    login(client, "user")
    first = upload(client, path).get_json()
    second = upload(client, path, "copy_of_horn.wav").get_json()
    assert first["ok"] and second["ok"]
    from src.models import AudioRecord, SystemAlert
    with app.app_context():
        rec = AudioRecord.query.filter_by(audio_id=second["record"]["audio_id"]).one()
        assert rec.duplicate_of is not None and rec.duplicate_of.audio_id == first["record"]["audio_id"]
        assert SystemAlert.query.filter(SystemAlert.kind.like("%duplicate%")).count() >= 1


@needs_model
def test_reencoded_copy_flagged_as_near_duplicate(app, client, no_gtm, tmp_path):
    y = impulse_in_noise(3.0, seed=5) + tone(3.0, 520, 0.3)
    a = write(tmp_path / "a.wav", y)
    b = write(tmp_path / "b.flac", np.clip(y * 0.6, -1, 1), subtype="PCM_24")  # re-encoded, quieter
    login(client, "user")
    first = upload(client, a).get_json()
    second = upload(client, b).get_json()
    from src.models import AudioRecord
    with app.app_context():
        rec = AudioRecord.query.filter_by(audio_id=second["record"]["audio_id"]).one()
        assert rec.duplicate_of is None
        assert rec.near_duplicate_of is not None and rec.near_duplicate_of.audio_id == first["record"]["audio_id"]


# --------------------------------------------------------------------------- #
# Live microphone
# --------------------------------------------------------------------------- #
def _wav_bytes(y, sr=16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_microphone_needs_consent_first(client):
    login(client, "security")
    r = client.post("/api/live/sessions", json={"window": 2})
    assert r.status_code == 400 and "notice" in r.get_json()["error"]


@needs_model
def test_live_session_status_windows_and_stop(app, client, no_gtm):
    login(client, "security")
    assert client.post("/api/live/consent").get_json()["ok"]
    r = client.post("/api/live/sessions", json={"window": 9, "hop": 0.1})  # clamped to 1-3 s windows
    assert r.status_code == 201
    s = r.get_json()
    assert s["window"] == 3.0 and s["hop"] >= 0.5
    code = s["session"]
    for status in ("paused", "active", "disconnected"):
        assert client.post(f"/api/live/sessions/{code}/status", json={"status": status}).get_json()["status"] == status
    assert client.post(f"/api/live/sessions/{code}/status", json={"status": "hacked"}).status_code == 400

    y = (impulse_in_noise(2.0, sr=16000, seed=7) * 0.9).astype(np.float32)

    def send(i):
        return client.post(f"/api/live/sessions/{code}/windows", data={
            "audio": (io.BytesIO(_wav_bytes(y)), "w.wav"), "meta": f'{{"window_index": {i}}}'},
            content_type="multipart/form-data")

    assert send(0).status_code == 200  # first window also loads the model (start-up warm-up in production)
    t = time.perf_counter()
    w = send(1)
    elapsed = time.perf_counter() - t
    assert w.status_code == 200, w.get_json()
    assert elapsed < 3.0, f"live window took {elapsed:.2f} s (SRS: under 3 s)"

    stop = client.post(f"/api/live/sessions/{code}/stop", json={"reason": "stopped"}).get_json()
    assert stop["ok"] and stop["windows"] >= 2
    assert client.post(f"/api/live/sessions/{code}/windows", data={
        "audio": (io.BytesIO(_wav_bytes(y)), "w.wav")}, content_type="multipart/form-data").status_code == 409
    from src.models import AuditLog
    with app.app_context():
        actions = {a.action for a in AuditLog.query.all()}
        assert {"MIC_CONSENT", "MIC_START", "MIC_STATUS"} <= actions


@needs_model
def test_live_results_are_smoothed_over_consecutive_windows_only(client, no_gtm):
    """Overlapping windows are averaged; silence, a pause or a gap starts over."""
    login(client, "security")
    client.post("/api/live/consent")
    code = client.post("/api/live/sessions", json={"window": 2, "hop": 1}).get_json()["session"]
    y = (impulse_in_noise(2.0, sr=16000, seed=7) * 0.9).astype(np.float32)

    def send(i, audio=None):
        return client.post(f"/api/live/sessions/{code}/windows", data={
            "audio": (io.BytesIO(_wav_bytes(y if audio is None else audio)), "w.wav"),
            "meta": f'{{"window_index": {i}}}'}, content_type="multipart/form-data").get_json()

    assert send(1)["smoothing_windows"] == 1  # nothing before it
    assert send(2)["smoothing_windows"] == 2
    assert send(3)["smoothing_windows"] == 3  # the configured maximum
    assert send(9)["smoothing_windows"] == 1, "a gap in the window indices must start a new stretch"
    assert send(10)["smoothing_windows"] == 2
    send(11, np.zeros(32000, dtype=np.float32))  # silence ends the stretch
    assert send(12)["smoothing_windows"] == 1
    client.post(f"/api/live/sessions/{code}/status", json={"status": "paused"})
    client.post(f"/api/live/sessions/{code}/status", json={"status": "active"})
    assert send(13)["smoothing_windows"] == 1, "a pause must start a new stretch"
    client.post(f"/api/live/sessions/{code}/stop", json={})


@needs_model
def test_silent_live_window_is_not_classified(client, no_gtm):
    login(client, "security")
    client.post("/api/live/consent")
    code = client.post("/api/live/sessions", json={"window": 2}).get_json()["session"]
    w = client.post(f"/api/live/sessions/{code}/windows", data={
        "audio": (io.BytesIO(_wav_bytes(np.zeros(32000, dtype=np.float32))), "w.wav"), "meta": "{}"},
        content_type="multipart/form-data").get_json()
    assert w["ok"]
    assert w.get("silent") or (w.get("record") or {}).get("final_category") in (None, "Unknown Sound")


def test_other_user_cannot_use_my_session(client):
    login(client, "security")
    client.post("/api/live/consent")
    r = client.post("/api/live/sessions", json={"window": 2})
    if r.status_code != 201:
        pytest.skip("Python model not available")
    code = r.get_json()["session"]
    client.post("/logout")
    login(client, "maintenance")
    assert client.post(f"/api/live/sessions/{code}/stop", json={}).status_code in (403, 404)
