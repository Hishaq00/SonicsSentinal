"""Small helpers: readable IDs, formatting filters, request helpers."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from flask import request
from sqlalchemy import update

from .extensions import db
from .models import IdCounter, utcnow

PASSWORD_RULES = "At least 8 characters with a letter and a number."


def next_code(prefix: str, dated: bool = True, width: int = 4) -> str:
    """Sequential, human-readable IDs such as SS-260924-0007 or USR-00012.

    The counter row is incremented inside the current transaction, so two
    concurrent requests can never receive the same number.
    """
    key = f"{prefix}-{utcnow():%y%m%d}" if dated else prefix
    counter = db.session.get(IdCounter, key)
    if counter is None:
        counter = IdCounter(key=key, value=0)
        db.session.add(counter)
        db.session.flush()
    db.session.execute(update(IdCounter).where(IdCounter.key == key).values(value=IdCounter.value + 1))
    db.session.flush()
    db.session.refresh(counter)
    return f"{key}-{counter.value:0{width}d}"


def valid_password(pw: str) -> bool:
    return len(pw or "") >= 8 and bool(re.search(r"[A-Za-z]", pw)) and bool(re.search(r"\d", pw))


def valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email or ""))


def client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or ""


def iso(dt: datetime | None) -> str:
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- filters --
def fmt_dt(dt: datetime | None, fmt: str = "%d %b %Y, %H:%M") -> str:
    return dt.strftime(fmt) + " UTC" if dt else "-"


def fmt_pct(x, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{float(x) * 100:.{digits}f}%"


def fmt_conf(x) -> str:
    return "-" if x is None else f"{float(x):.2f}"


def fmt_seconds(x) -> str:
    if x is None:
        return "-"
    x = float(x)
    if x < 60:
        return f"{x:.1f} s"
    m, s = divmod(int(round(x)), 60)
    return f"{m}:{s:02d}"


def fmt_age(x) -> str:
    """Long waiting times in the largest sensible unit (e.g. "45 min", "7 h", "128 days")."""
    if x is None:
        return "-"
    x = float(x)
    if x < 3600:
        return f"{max(1, round(x / 60))} min"
    if x < 48 * 3600:
        return f"{round(x / 3600)} h"
    return f"{round(x / 86400)} days"


def fmt_timecode(x) -> str:
    if x is None:
        return "-"
    x = max(0.0, float(x))
    m, s = divmod(x, 60)
    return f"{int(m)}:{s:04.1f}"


def fmt_bytes(n) -> str:
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def register_filters(app) -> None:
    app.jinja_env.filters.update(
        dt=fmt_dt, pct=fmt_pct, conf=fmt_conf, secs=fmt_seconds, age=fmt_age, tc=fmt_timecode,
        bytes=fmt_bytes, iso=iso, slug=slug,
    )
