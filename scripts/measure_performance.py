"""Measure the SRS 1.7-1 performance targets and write ``reports/performance.md``.

    python -m scripts.measure_performance            # 5 runs per measurement

* Upload: a 30-second clip must be classified within 8 seconds.
  Measured = server pipeline (validation, pre-processing, quality, waveform and
  spectrogram images, 193 features, Python model, rule engine, database) +
  the GTM model on the same segments. In the app GTM runs in the browser
  (TF.js); here the exported model is run with the NumPy runner in
  ``gtm_model/inference.py`` as a stand-in for that time.
* Live: one 2-second microphone window must produce a prediction within 3 s.

Runs against a throw-away database and storage folder; the model is loaded
once before timing (the app does this at start-up, ``WARMUP_MODELS``).
"""
from __future__ import annotations

import argparse
import platform
import shutil
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from config import REPORTS_DIR


def _clip(seconds: float, sr: int, seed: int) -> np.ndarray:
    """Background noise with a few impulsive and tonal events, like a real 30 s recording."""
    rng = np.random.default_rng(seed)
    y = 0.02 * rng.standard_normal(int(seconds * sr))
    t = np.arange(int(0.6 * sr)) / sr
    for start in rng.uniform(0.5, seconds - 1.0, 6):
        s = int(start * sr)
        burst = rng.standard_normal(len(t)) * np.exp(-t / 0.05) * 0.7 + 0.3 * np.sin(2 * np.pi * 900 * t) * np.exp(-t / 0.3)
        y[s:s + len(t)] += burst[: len(y) - s]
    return np.clip(y, -1, 1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    from gtm_model import pipeline as gtm_pipeline
    from gtm_model.inference import GTMClassifier
    from src import create_app
    from src.app_config import TestConfig
    from src.extensions import db
    from src.models import User
    from src.services import live_service, pipeline
    from src.services.settings_service import update_settings
    from python_models.inference import get_classifier

    work = Path(tempfile.mkdtemp(prefix="ss-perf-"))

    class Cfg(TestConfig):
        SQLALCHEMY_DATABASE_URI = "sqlite:///" + (work / "perf.db").as_posix()
        STORAGE_DIR = str(work / "storage")
        SEED_DEMO_USERS = True

    app = create_app(Cfg)
    gtm = GTMClassifier()
    upload_s, server_s, gtm_s, live_s = [], [], [], []
    with app.app_context():
        update_settings({"gtm": {"enabled": False}}, "perf")  # GTM is timed separately below
        db.session.commit()
        user = User.query.filter_by(username="security").one()
        clf = get_classifier()
        if not clf.available:
            raise SystemExit("No trained Python model (python_models/saved/active.json).")
        # warm-up: model load and first-call overheads, not part of the measurement
        warm = work / "warm.wav"
        sf.write(warm, _clip(3.0, 22050, 0), 22050)
        pipeline.analyze_file(warm, "warm.wav", user)

        for i in range(args.runs):
            path = work / f"clip{i}.wav"
            sf.write(path, _clip(30.0, 44100, i + 1), 44100, subtype="PCM_16")
            t0 = time.perf_counter()
            record = pipeline.analyze_file(path, f"clip{i}.wav", user)
            t1 = time.perf_counter()
            audio = gtm_pipeline.load(path, "upload")
            segs = [s for s, _ in gtm_pipeline.app_segments(audio)]
            gtm.predict_inputs([gtm_pipeline.segment_input(audio, s)[0] for s in segs])
            t2 = time.perf_counter()
            assert record.final_category, "clip was not classified"
            server_s.append(t1 - t0)
            gtm_s.append(t2 - t1)
            upload_s.append(t2 - t0)

        session = live_service.start_session(user, 2.0, 1.0)
        import io
        for i in range(args.runs + 1):
            buf = io.BytesIO()
            sf.write(buf, _clip(2.0, 16000, 100 + i), 16000, format="WAV", subtype="PCM_16")
            t0 = time.perf_counter()
            live_service.process_window(session, buf.getvalue(), None, i, None)
            if i:  # the first window initialises the session state
                live_s.append(time.perf_counter() - t0)
        live_service.stop_session(session)
        db.session.remove()
        db.engine.dispose()
    shutil.rmtree(work, ignore_errors=True)

    def row(name, xs, target):
        return (f"| {name} | {statistics.median(xs):.2f} s | {max(xs):.2f} s | {target} | "
                f"{'met' if max(xs) <= float(target.split()[1]) else 'NOT met'} |")

    lines = [
        "# Performance (SRS 1.7-1)",
        "",
        f"Measured {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC with `python -m scripts.measure_performance "
        f"--runs {args.runs}` on {platform.platform()}, Python {platform.python_version()}, "
        f"Python model `{clf.version}`, GTM `{gtm.version}`.",
        "",
        "| Measurement | Median | Worst | Target | Result |",
        "|---|---|---|---|---|",
        row("30 s upload: server pipeline", server_s, "<= 8 s"),
        row("30 s upload: GTM on the same segments (NumPy runner)", gtm_s, "<= 8 s"),
        row("30 s upload: total (server + GTM)", upload_s, "<= 8 s"),
        row("Live 2 s window: server prediction", live_s, "<= 3 s"),
        "",
        "Server pipeline = validation, pre-processing, quality analysis, waveform and spectrogram images, features, "
        "Python model, rule engine and database writes. In the web app the GTM model runs in the browser at the same "
        "time as the server works, so the user-visible time is at most the total above. Times exclude the one-time "
        "model load at start-up.",
    ]
    out = REPORTS_DIR / "performance.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
