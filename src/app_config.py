"""Flask configuration. Values can be overridden with environment variables
(see ``.env.example``)."""
from __future__ import annotations

import os
import secrets
from datetime import timedelta

from config import DATABASE_DIR, STORAGE_DIR


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or f"sqlite:///{(DATABASE_DIR / 'sonicsentinel.db').as_posix()}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    STORAGE_DIR = os.environ.get("STORAGE_DIR") or str(STORAGE_DIR)
    # One file per request (batch uploads send files one after another).
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "30")) * 1024 * 1024
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _bool("SESSION_COOKIE_SECURE", False)
    REMEMBER_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=int(os.environ.get("SESSION_HOURS", "8")))
    WTF_CSRF_TIME_LIMIT = None
    GTM_MODEL_URL = os.environ.get("GTM_MODEL_URL", "")
    WARMUP_MODELS = _bool("WARMUP_MODELS", True)
    SEED_DEMO_USERS = _bool("SEED_DEMO_USERS", True)
    APP_NAME = "SonicSentinel AI"


class TestConfig(Config):
    TESTING = True
    SECRET_KEY = "test-secret"
    WTF_CSRF_ENABLED = False
    WARMUP_MODELS = False
    SEED_DEMO_USERS = False
