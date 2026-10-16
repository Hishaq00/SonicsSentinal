"""SonicSentinel AI - Flask application factory."""
from __future__ import annotations

import logging
import os
import threading
import time

import click
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user
from flask_wtf.csrf import CSRFError
from sqlalchemy import event
from sqlalchemy.engine import Engine

from config import DATABASE_DIR, PROJECT_ROOT, SEVERITY_LEVELS, STORAGE_DIR, class_info

from .app_config import Config
from .extensions import csrf, db, login_manager

__version__ = "1.0.3"
log = logging.getLogger("sonicsentinel")

DEMO_USERS = [
    # username, password, role, full name
    ("admin", "Admin@2026", "admin", "Site Administrator"),
    ("evaluator", "Evaluate@2026", "admin", "Competition Evaluator"),
    ("reviewer", "Review@2026", "reviewer", "Audio Reviewer"),
    ("security", "Secure@2026", "security", "Security Operator"),
    ("maintenance", "Maintain@2026", "maintenance", "Maintenance Operator"),
    ("user", "User@2026", "user", "Normal User"),
]


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, connection_record):  # pragma: no cover - driver specific
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()


def create_app(config_class: type = Config) -> Flask:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"),
                static_folder=str(PROJECT_ROOT / "static"), static_url_path="/static")
    app.config.from_object(config_class)
    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite"):
        app.config.setdefault("SQLALCHEMY_ENGINE_OPTIONS", {"connect_args": {"check_same_thread": False, "timeout": 15}})

    db.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to continue."
    login_manager.login_message_category = "info"

    from .models import User

    @login_manager.user_loader
    def load_user(user_id: str):
        return db.session.get(User, int(user_id))

    from .blueprints import admin, alerts, analysis, analytics, auth, gtm, history, live, main, review

    for bp in (auth.bp, main.bp, analysis.bp, live.bp, alerts.bp, review.bp, history.bp, analytics.bp, admin.bp, gtm.bp):
        app.register_blueprint(bp)

    from .utils import register_filters

    register_filters(app)
    _register_context(app)
    _register_errors(app)
    _register_hooks(app)
    _register_cli(app)

    with app.app_context():
        db.create_all()
        if app.config.get("SEED_DEMO_USERS"):
            seed_users()
        try:
            admin.sync_python_versions("system")
            db.session.commit()
        except Exception:  # model folder may be empty on a fresh install
            db.session.rollback()

    if app.config.get("WARMUP_MODELS") and os.environ.get("WERKZEUG_RUN_MAIN", "true") == "true":
        threading.Thread(target=_warm_up, args=(app,), daemon=True).start()
    return app


def seed_users() -> None:
    from .models import User
    from .utils import next_code

    if User.query.count():
        return
    for username, password, role, name in DEMO_USERS:
        u = User(user_code=next_code("USR", dated=False, width=5), username=username,
                 email=f"{username}@sonicsentinel.local", full_name=name, role=role, organization="Demo site",
                 department={"admin": "IT", "reviewer": "Audio Review", "security": "Security",
                             "maintenance": "Maintenance", "user": "Operations"}[role])
        u.set_password(password)
        db.session.add(u)
    db.session.commit()
    log.info("Demo accounts created (see README).")


def _warm_up(app: Flask) -> None:
    """Load librosa / numba and the model once so the first request is fast."""
    time.sleep(0.5)
    with app.app_context():
        try:
            import numpy as np

            from python_models.inference import get_classifier

            clf = get_classifier()
            if clf.available:
                clf.predict_segments([np.zeros(44100, dtype=np.float32) + 1e-3])
            log.info("Model warm-up complete.")
        except Exception as exc:  # pragma: no cover
            log.warning("Warm-up skipped: %s", exc)


DASHBOARD_LABELS = {"reviewer": "Review desk", "security": "Security console", "maintenance": "Maintenance console"}


def _nav(user) -> list[dict]:
    from .security import can

    if not user.is_authenticated:
        return []
    sections = [
        {"title": "Monitor", "items": [
            {"endpoint": "main.dashboard", "label": DASHBOARD_LABELS.get(user.role, "Overview"), "icon": "grid",
             "hide_for": {"admin"}},
            {"endpoint": "main.overview", "label": "Site overview", "icon": "gauge", "only_for": {"reviewer", "security", "maintenance"}},
            {"endpoint": "live.monitor", "label": "Live monitor", "icon": "mic", "perm": "live_monitor"},
            {"endpoint": "analysis.analyze", "label": "Analyze audio", "icon": "upload", "perm": "upload"},
            {"endpoint": "alerts.list_alerts", "label": "Alerts", "icon": "bell", "badge": "alerts"},
        ]},
        {"title": "Investigate", "items": [
            {"endpoint": "history.history", "label": "Event history", "icon": "list"},
            {"endpoint": "main.timeline", "label": "Timeline", "icon": "clock"},
            {"endpoint": "review.queue", "label": "Review queue", "icon": "check", "perm": "review", "badge": "review"},
            {"endpoint": "analytics.index", "label": "Analytics", "icon": "chart", "perm": "analytics"},
        ]},
        {"title": "Administration", "items": [
            {"endpoint": "admin.dashboard", "label": "System overview", "icon": "gauge", "perm": "admin"},
            {"endpoint": "admin.users", "label": "Users & roles", "icon": "users", "perm": "admin"},
            {"endpoint": "admin.rules", "label": "Alert rules", "icon": "sliders", "perm": "admin"},
            {"endpoint": "admin.settings", "label": "Settings", "icon": "settings", "perm": "admin"},
            {"endpoint": "admin.models", "label": "Models", "icon": "cpu", "perm": "admin"},
            {"endpoint": "admin.evaluation", "label": "Model comparison", "icon": "scale", "perm": "evaluation"},
            {"endpoint": "admin.dataset", "label": "Dataset", "icon": "database", "perm": "admin"},
            {"endpoint": "admin.audit_trail", "label": "Audit trail", "icon": "file", "perm": "admin"},
            {"endpoint": "admin.system_alerts", "label": "System alerts", "icon": "alert", "perm": "admin",
             "badge": "system"},
        ]},
    ]
    out = []
    for s in sections:
        items = [i for i in s["items"] if ("perm" not in i or can(user, i["perm"]))
                 and user.role not in i.get("hide_for", ())
                 and ("only_for" not in i or user.role in i["only_for"])]
        if items:
            out.append({"title": s["title"], "items": items})
    return out


NAV_MAP = {
    "live.": "live.monitor",
    "alerts.": "alerts.list_alerts",
    "history.": "history.history",
    "review.": "review.queue",
    "analytics.": "analytics.index",
    "main.timeline": "main.timeline",
    "main.dashboard": "main.dashboard",
    "main.overview": "main.overview",
    "analysis.record_detail": "history.history",
    "analysis.": "analysis.analyze",
    "admin.dashboard": "admin.dashboard",
    "admin.users": "admin.users",
    "admin.update_user": "admin.users",
    "admin.rules": "admin.rules",
    "admin.settings": "admin.settings",
    "admin.evaluation": "admin.evaluation",
    "admin.dataset": "admin.dataset",
    "admin.audit": "admin.audit_trail",
    "admin.system": "admin.system_alerts",
    "admin.": "admin.models",
}


def _active_nav(endpoint: str | None) -> str | None:
    """Map the current endpoint to the sidebar item that should be highlighted."""
    if not endpoint:
        return None
    for prefix, target in NAV_MAP.items():
        if endpoint.startswith(prefix):
            return target
    return None


def _badges(user) -> dict:
    from .models import Alert, AudioRecord, SystemAlert
    from .security import can
    from .services.alerts_service import visible_alerts

    if not user.is_authenticated:
        return {}
    out = {"alerts": visible_alerts(user).filter(Alert.status.in_(("Active", "Escalated"))).count()}
    if can(user, "review"):
        out["review"] = AudioRecord.query.filter(AudioRecord.needs_review.is_(True),
                                                 AudioRecord.is_evaluation.is_(False)).count()
    if can(user, "admin"):
        out["system"] = SystemAlert.query.filter_by(resolved=False).count()
    return out


def _register_context(app: Flask) -> None:
    from .security import can

    @app.context_processor
    def inject():
        info = class_info()
        return {
            "app_name": app.config["APP_NAME"],
            "app_version": __version__,
            "can": lambda perm: can(current_user, perm),
            "nav_sections": _nav(current_user) if current_user else [],
            "badges": _badges(current_user) if current_user and current_user.is_authenticated else {},
            "class_codes": {k: v.get("code", "") for k, v in info.items()},
            # Parent category of a class, e.g. "Animal Sound" for "Dog Bark", so a detection can be
            # shown as the species without losing the category it belongs to.
            "class_parents": {k: v["parent"] for k, v in info.items() if v.get("parent")},
            "nav_active": _active_nav(request.endpoint),
            "severity_levels": SEVERITY_LEVELS,
            "user_theme": (current_user.pref("theme", "system") if current_user and current_user.is_authenticated else "system"),
            "alert_sound": bool(current_user.pref("alert_sound", False)) if current_user and current_user.is_authenticated else False,
        }


def _wants_json() -> bool:
    return request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json"


def _register_errors(app: Flask) -> None:
    messages = {
        400: ("Bad request", "The request could not be understood."),
        401: ("Sign in required", "Please sign in to continue."),
        403: ("Access denied", "Your role does not have permission to open this page."),
        404: ("Not found", "The page or record does not exist, or it was removed by the retention policy."),
        405: ("Method not allowed", "This action is not allowed here."),
        413: ("File too large", "The uploaded file is larger than the allowed size."),
        429: ("Too many attempts", "Please wait a little and try again."),
        500: ("Something went wrong", "An unexpected error occurred. The error has been logged."),
    }

    def handler(code):
        def _h(err):
            if code == 500:
                db.session.rollback()
                log.exception("Unhandled error: %s", err)
            title, text = messages[code]
            if _wants_json():
                return jsonify(ok=False, error=text), code
            return render_template("errors/error.html", code=code, title=title, text=text), code
        return _h

    for code in messages:
        app.register_error_handler(code, handler(code))

    @app.errorhandler(CSRFError)
    def csrf_error(err):
        if _wants_json():
            return jsonify(ok=False, error="Security token expired. Reload the page."), 400
        return render_template("errors/error.html", code=400, title="Security check failed",
                               text="Your session security token has expired. Go back, reload the page and try again."), 400


def _register_hooks(app: Flask) -> None:
    state = {"last": 0.0}

    @app.before_request
    def enforce_authentication():
        """Ensure unauthenticated users can only view the landing page and public auth pages."""
        if app.testing:
            return
        # Whitelist static assets
        if request.path.startswith("/static"):
            return
        # Whitelist landing page endpoints
        if request.endpoint in ("main.index",):
            return
        # Whitelist public auth endpoints
        if request.endpoint in ("auth.login", "auth.register", "auth.privacy"):
            return
        # If user is not authenticated, block access
        if not current_user.is_authenticated:
            if _wants_json():
                return jsonify(ok=False, error="Authentication required. Please sign in to perform this action."), 401
            flash("You must be signed in before taking any action. Please sign in to continue.", "info")
            return redirect(url_for("auth.login", next=request.path))

    @app.before_request
    def maintenance():
        # Lightweight scheduler: finalise abandoned GTM jobs and apply retention (at most every 5 min).
        if app.testing or request.path.startswith("/static"):
            return
        now = time.time()
        if now - state["last"] < 300:
            return
        state["last"] = now
        try:
            from .services import pipeline, retention

            pipeline.finalize_stale(minutes=5)
            retention.run_if_due()
        except Exception as exc:  # never block a request because of housekeeping
            db.session.rollback()
            log.warning("Maintenance task failed: %s", exc)

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        resp.headers.setdefault("Permissions-Policy", "microphone=(self), camera=(), geolocation=()")
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self'; worker-src 'self' blob:; "
            "connect-src 'self' https://teachablemachine.withgoogle.com https://storage.googleapis.com; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        )
        return resp


def _register_cli(app: Flask) -> None:
    @app.cli.command("init-db")
    def init_db():
        """Create tables and demo accounts."""
        db.create_all()
        seed_users()
        click.echo("Database ready.")

    @app.cli.command("create-admin")
    @click.option("--username", required=True)
    @click.option("--email", required=True)
    @click.option("--password", required=True)
    def create_admin(username, email, password):
        """Create an administrator account."""
        from .models import User
        from .utils import next_code, valid_password

        if not valid_password(password):
            raise click.BadParameter("password needs 8+ characters with a letter and a number")
        u = User(user_code=next_code("USR", dated=False, width=5), username=username, email=email,
                 full_name=username, role="admin")
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        click.echo(f"Administrator {username} created.")

    @app.cli.command("purge")
    def purge():
        """Apply the data-retention policy now."""
        from .services import retention

        click.echo(retention.run_retention("cli"))

    @app.cli.command("export-schema")
    def export_schema():
        """Write database/schema.sql from the SQLAlchemy models."""
        from sqlalchemy.schema import CreateIndex, CreateTable

        lines = ["-- SonicSentinel AI database schema (generated from src/models.py)", ""]
        for table in db.metadata.sorted_tables:
            lines.append(str(CreateTable(table).compile(db.engine)).strip() + ";")
            for idx in table.indexes:
                lines.append(str(CreateIndex(idx).compile(db.engine)).strip() + ";")
            lines.append("")
        path = PROJECT_ROOT / "database" / "schema.sql"
        path.write_text("\n".join(lines), encoding="utf-8")
        click.echo(f"Schema written to {path}")
