"""Role-based access control.

Roles: Normal user, Audio reviewer, Security operator, Maintenance operator,
Administrator. Permissions are checked on every route with
``@permission_required`` and in templates with ``can('permission')``.
Alert actions are additionally scoped by category: each alert rule lists the
roles responsible for it (e.g. Machinery Fault -> maintenance).
"""
from __future__ import annotations

from functools import wraps

from flask import abort
from flask_login import current_user, login_required

ALL_ROLES = {"user", "reviewer", "security", "maintenance", "admin"}
OPERATORS = {"reviewer", "security", "maintenance", "admin"}

PERMISSIONS: dict[str, set[str]] = {
    "upload": ALL_ROLES,
    "live_monitor": ALL_ROLES,
    "batch_upload": OPERATORS,
    "view_all_records": OPERATORS,
    "review": {"reviewer", "admin"},
    "close_event": OPERATORS,
    "act_on_alerts": {"security", "maintenance", "admin"},
    "analytics": OPERATORS,
    "export": {"admin"},
    "admin": {"admin"},
    "evaluation": {"admin"},
}

PERMISSION_LABELS = {
    "upload": "Upload and analyse audio",
    "live_monitor": "Live microphone monitoring",
    "batch_upload": "Batch upload",
    "view_all_records": "View all events",
    "review": "Manual review and overrides",
    "close_event": "Close events",
    "act_on_alerts": "Acknowledge / dismiss / escalate alerts",
    "analytics": "Analytics and reports",
    "export": "Export CSV / Excel",
    "admin": "Users, settings, rules, models, audit",
    "evaluation": "Model comparison runs",
}


def can(user, permission: str) -> bool:
    if user is None or not getattr(user, "is_authenticated", False):
        return False
    return user.role in PERMISSIONS.get(permission, set())


def can_view_record(user, record) -> bool:
    return bool(user and user.is_authenticated and (record.user_id == user.id or can(user, "view_all_records")))


def can_act_on_alert(user, alert) -> bool:
    if not can(user, "act_on_alerts"):
        return False
    if user.role == "admin":
        return True
    return user.role in (alert.responsible_roles or [])


def permission_required(permission: str):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapper(*args, **kwargs):
            if not can(current_user, permission):
                abort(403)
            return view(*args, **kwargs)

        return wrapper

    return decorator


def role_matrix() -> list[dict]:
    """Permission matrix for the admin Users page and the documentation."""
    return [{"permission": p, "label": PERMISSION_LABELS[p], "roles": sorted(PERMISSIONS[p])} for p in PERMISSIONS]
