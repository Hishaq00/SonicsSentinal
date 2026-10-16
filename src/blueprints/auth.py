"""Registration, login / logout, profile management and the privacy notice."""
from __future__ import annotations

from datetime import timedelta
from urllib.parse import urlparse

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import func

from ..extensions import db
from ..models import ROLES, User, utcnow
from ..services import audit, monitoring
from ..services.settings_service import get_settings
from ..utils import PASSWORD_RULES, client_ip, next_code, valid_email, valid_password

bp = Blueprint("auth", __name__)


def _safe_next(target: str | None) -> str | None:
    if not target:
        return None
    parsed = urlparse(target)
    if parsed.netloc or parsed.scheme or not target.startswith("/"):
        return None
    return target


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        ident = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        user = User.query.filter(or_ident(ident)).first() if ident else None
        sec = get_settings()["security"]
        if user and user.locked_until and user.locked_until > utcnow():
            minutes = int((user.locked_until - utcnow()).total_seconds() // 60) + 1
            flash(f"Too many failed attempts. The account is locked for {minutes} more minute(s).", "error")
            audit.log("LOGIN_FAILED", "user", ident, {"reason": "locked"}, username=ident[:40], commit=True)
            return render_template("auth/login.html", username=ident), 429
        if user and user.check_password(password) and user.is_active:
            user.failed_logins = 0
            user.locked_until = None
            user.last_login_at = utcnow()
            login_user(user, remember=bool(request.form.get("remember")))
            audit.log("LOGIN", "user", user.user_code, {"remember": bool(request.form.get("remember"))}, user=user)
            db.session.commit()
            return redirect(_safe_next(request.args.get("next")) or url_for("main.dashboard"))
        reason = "disabled account" if (user and not user.is_active) else "invalid credentials"
        if user and user.is_active:
            user.failed_logins = (user.failed_logins or 0) + 1
            if user.failed_logins >= int(sec["max_failed_logins"]):
                user.locked_until = utcnow() + timedelta(minutes=int(sec["lockout_minutes"]))
                user.failed_logins = 0
                reason = "locked after repeated failures"
        audit.log("LOGIN_FAILED", "user", ident[:40], {"reason": reason}, username=ident[:40])
        monitoring.on_failed_login(ident[:40], client_ip())
        db.session.commit()
        flash("The username or password is incorrect." if reason != "disabled account"
              else "This account has been disabled. Contact an administrator.", "error")
        return render_template("auth/login.html", username=ident), 401
    return render_template("auth/login.html", username="")


def or_ident(ident: str):
    return (func.lower(User.username) == ident.lower()) | (func.lower(User.email) == ident.lower())


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = {k: (request.form.get(k) or "").strip() for k in
            ("full_name", "username", "email", "organization", "department", "phone", "requested_role")}
    if request.method == "POST":
        errors = []
        password = request.form.get("password") or ""
        if len(form["full_name"]) < 2:
            errors.append("Enter your full name.")
        if not form["username"] or not form["username"].replace("_", "").replace(".", "").isalnum() or len(form["username"]) < 3:
            errors.append("Username must be at least 3 letters or numbers (dots and underscores allowed).")
        if not valid_email(form["email"]):
            errors.append("Enter a valid email address.")
        if not valid_password(password):
            errors.append(f"Password: {PASSWORD_RULES}")
        if password != (request.form.get("confirm") or ""):
            errors.append("The passwords do not match.")
        if not request.form.get("consent"):
            errors.append("Please accept the privacy notice.")
        if User.query.filter(func.lower(User.username) == form["username"].lower()).first():
            errors.append("That username is already taken.")
        if User.query.filter(func.lower(User.email) == form["email"].lower()).first():
            errors.append("An account with that email already exists.")
        requested = form["requested_role"] if form["requested_role"] in ROLES else "user"
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("auth/register.html", form=form, roles=ROLES), 400
        user = User(user_code=next_code("USR", dated=False, width=5), username=form["username"],
                    email=form["email"].lower(), full_name=form["full_name"], organization=form["organization"],
                    department=form["department"], phone=form["phone"], role="user",
                    preferences={"requested_role": requested} if requested != "user" else {})
        user.set_password(password)
        db.session.add(user)
        db.session.flush()
        audit.log("REGISTER", "user", user.user_code, {"requested_role": requested}, user=user)
        db.session.commit()
        login_user(user)
        msg = "Welcome to SonicSentinel AI."
        if requested != "user":
            msg += f" Your request for the {ROLES[requested]} role has been sent to an administrator."
        flash(msg, "success")
        return redirect(url_for("main.dashboard"))
    return render_template("auth/register.html", form=form, roles=ROLES)


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    audit.log("LOGOUT", "user", current_user.user_code)
    db.session.commit()
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user
    if request.method == "POST":
        errors = []
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        if len(full_name) < 2:
            errors.append("Enter your full name.")
        if not valid_email(email):
            errors.append("Enter a valid email address.")
        elif User.query.filter(func.lower(User.email) == email, User.id != user.id).first():
            errors.append("Another account uses that email.")
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("auth/profile.html"), 400
        before = {"full_name": user.full_name, "email": user.email}
        user.full_name = full_name
        user.email = email
        for key in ("phone", "organization", "department", "job_title"):
            setattr(user, key, (request.form.get(key) or "").strip()[:120])
        prefs = dict(user.preferences or {})
        prefs["theme"] = request.form.get("theme") if request.form.get("theme") in ("light", "dark", "system") else "system"
        prefs["alert_sound"] = bool(request.form.get("alert_sound"))
        user.preferences = prefs
        audit.log("PROFILE_UPDATE", "user", user.user_code, {"before": before, "after": {"full_name": full_name, "email": email}})
        db.session.commit()
        flash("Profile saved.", "success")
        return redirect(url_for("auth.profile"))
    return render_template("auth/profile.html")


@bp.route("/profile/password", methods=["POST"])
@login_required
def change_password():
    user = current_user
    current = request.form.get("current_password") or ""
    new = request.form.get("new_password") or ""
    if not user.check_password(current):
        flash("Your current password is incorrect.", "error")
    elif not valid_password(new):
        flash(f"New password: {PASSWORD_RULES}", "error")
    elif new != (request.form.get("confirm_password") or ""):
        flash("The new passwords do not match.", "error")
    else:
        user.set_password(new)
        audit.log("PASSWORD_CHANGE", "user", user.user_code)
        db.session.commit()
        flash("Password changed.", "success")
    return redirect(url_for("auth.profile"))


@bp.route("/privacy")
def privacy():
    return render_template("auth/privacy.html", settings=get_settings())
