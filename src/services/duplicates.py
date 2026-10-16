"""Duplicate (SHA-256) and near-duplicate (acoustic fingerprint) detection."""
from __future__ import annotations

import numpy as np

from audio_preprocessing.fingerprint import NEAR_DUPLICATE_BER, compare_fingerprints, descriptor_from_bytes

from ..extensions import db
from ..models import AudioRecord

SHORTLIST = 25
MIN_SIMILARITY = 0.85


def _uploads():
    return AudioRecord.query.filter(AudioRecord.source.in_(("upload", "batch")), AudioRecord.is_evaluation.is_(False))


def find_exact(sha256: str, exclude_id: int | None = None) -> AudioRecord | None:
    q = _uploads().filter(AudioRecord.sha256 == sha256)
    if exclude_id:
        q = q.filter(AudioRecord.id != exclude_id)
    return q.order_by(AudioRecord.created_at).first()


def find_near(fingerprint: bytes, descriptor: bytes, exclude_id: int | None = None) -> tuple[AudioRecord | None, float | None]:
    """Shortlist by spectral-descriptor similarity, then confirm with the bit error rate."""
    q = db.session.query(AudioRecord.id, AudioRecord.fp_descriptor).filter(
        AudioRecord.fp_descriptor.isnot(None),
        AudioRecord.source.in_(("upload", "batch")),
        AudioRecord.is_evaluation.is_(False),
    )
    if exclude_id:
        q = q.filter(AudioRecord.id != exclude_id)
    rows = q.all()
    if not rows:
        return None, None
    target = descriptor_from_bytes(descriptor)
    mat = np.stack([descriptor_from_bytes(r.fp_descriptor) for r in rows])
    sims = mat @ target
    order = np.argsort(-sims)[:SHORTLIST]
    best, best_ber = None, 1.0
    for i in order:
        if sims[i] < MIN_SIMILARITY:
            break
        rec = db.session.get(AudioRecord, rows[i].id)
        if rec is None or rec.fingerprint is None:
            continue
        ber, _ = compare_fingerprints(fingerprint, rec.fingerprint)
        if ber < best_ber:
            best, best_ber = rec, ber
    if best is not None and best_ber <= NEAR_DUPLICATE_BER:
        return best, best_ber
    return None, None
