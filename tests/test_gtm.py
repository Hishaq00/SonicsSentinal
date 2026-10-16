"""Google Teachable Machine integration tests.

* The Python reference implementation (feature_extraction/gtm_spectrogram.py,
  gtm_model/pipeline.py) and the browser implementation (static/js/gtm.js) must
  give the same numbers: spectrogram, dither, window choice, padding and
  normalisation (checked with Node.js when it is installed).
* The exported GTM model gives the same probabilities in TensorFlow.js (the
  app's own tf.min.js) as the Python runner used for training and reports,
  end to end through gtm.js.
* Label checks and GTM score validation.
* GTM model files are served, and GTM results posted by the browser finalise
  a record (using the format-compatible test fixture, NOT a real GTM model).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from config import GTM_MODEL_DIR, PROJECT_ROOT, class_names
from feature_extraction.gtm_spectrogram import (
    GTM_BINS,
    GTM_FRAMES,
    GTM_INPUT_SAMPLES,
    best_window_start,
    fit_input,
    gtm_spectrogram,
    normalize,
)

from .conftest import impulse_in_noise, login, tone, upload

GTM_JS = PROJECT_ROOT / "static" / "js" / "gtm.js"
TF_JS = PROJECT_ROOT / "static" / "vendor" / "tf.min.js"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
needs_gtm_model = pytest.mark.skipif(not (GTM_MODEL_DIR / "model.json").exists(), reason="no exported GTM model")


def run_node(script: str, timeout: int = 180) -> str:
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_gtm_spectrogram_shape_and_scale():
    x = tone(1.1, 1000, 0.5, sr=44100)
    spec = gtm_spectrogram(x)
    assert spec.shape == (GTM_FRAMES, GTM_BINS) == (43, 232)
    # 1 kHz sits in bin round(1000 / (44100 / 2048)) = 46
    assert int(np.argmax(spec.mean(axis=0))) in (45, 46, 47)
    z = normalize(spec)
    assert abs(z.mean()) < 1e-4 and abs(z.std() - 1) < 1e-3


def test_best_window_is_centred_on_the_event():
    y = impulse_in_noise(3.0, sr=44100, at=2.2)
    start = best_window_start(y, 44100)
    assert start <= int(2.2 * 44100) <= start + GTM_INPUT_SAMPLES


@needs_node
@pytest.mark.parametrize("dither", [False, True])
def test_python_and_javascript_spectrograms_match(tmp_path, dither):
    rng = np.random.default_rng(3)
    x = (0.3 * np.sin(2 * np.pi * 700 * np.arange(GTM_INPUT_SAMPLES) / 44100) + 0.05 * rng.standard_normal(GTM_INPUT_SAMPLES)).astype(np.float32)
    x[30000:] = 0.0  # digital silence: only the dither keeps these frames finite
    (tmp_path / "x.json").write_text(json.dumps(x.tolist()))
    script = f"""
      const fs = require('fs'); const vm = require('vm');
      const ctx = {{ window: {{}}, Math, Float32Array, Float64Array, Uint32Array, console }};
      vm.createContext(ctx);
      vm.runInContext(fs.readFileSync({json.dumps(str(GTM_JS))}, 'utf8'), ctx);
      const x = Float32Array.from(JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / 'x.json'))}, 'utf8')));
      const spec = ctx.window.SSGTM.spectrogram(x, {json.dumps(dither)});
      process.stdout.write(JSON.stringify({{spec: Array.from(spec), norm: Array.from(ctx.window.SSGTM.normalize(spec))}}));
    """
    out = json.loads(run_node(script))
    js = np.array(out["spec"], dtype=np.float32).reshape(GTM_FRAMES, GTM_BINS)
    py = gtm_spectrogram(x, dither=dither)
    assert np.max(np.abs(js - py)) < 0.01  # dB, including the dither-only frames
    js_norm = np.array(out["norm"], dtype=np.float32).reshape(GTM_FRAMES, GTM_BINS)
    assert np.max(np.abs(js_norm - normalize(py))) < 1e-3


@needs_node
def test_python_and_javascript_pick_the_same_window(tmp_path):
    """bestWindowStart (incl. Math.round on .5), fitInput centring and classifySegment's sample range."""
    rng = np.random.default_rng(11)
    cases = []
    for n, at in ((88200, 0.3), (88200, 1.7), (60000, 0.9), (45056, 0.5), (30000, 0.4), (1200, 0.0), (132300, 2.8)):
        y = (0.01 * rng.standard_normal(n)).astype(np.float32)
        k = int(at * 44100)
        y[k:k + 2000] += 0.8
        cases.append(y)
    (tmp_path / "cases.json").write_text(json.dumps([c.tolist() for c in cases]))
    segments = [(0.0, 2.0), (1.0, 3.0), (0.345, 2.345), (2.999, 5.0)]
    script = f"""
      const fs = require('fs'); const vm = require('vm');
      const ctx = {{ window: {{}}, Math, Float32Array, Float64Array, Uint32Array, console }};
      vm.createContext(ctx);
      vm.runInContext(fs.readFileSync({json.dumps(str(GTM_JS))}, 'utf8'), ctx);
      const G = ctx.window.SSGTM;
      const cases = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / 'cases.json'))}, 'utf8')).map(c => Float32Array.from(c));
      const out = cases.map(y => {{ const ws = G.bestWindowStart(y); const w = G.fitInput(y.subarray(ws, ws + G.INPUT));
                                    let first = 0; while (first < w.length && w[first] === 0) first++;
                                    return {{ ws, first, len: w.length }}; }});
      const long = cases[cases.length - 1];
      const bounds = {json.dumps(segments)}.map(([a, b]) => {{
        const s0 = Math.max(0, Math.floor(a * G.P.SR)); const s1 = Math.min(long.length, Math.max(s0 + 1, Math.ceil(b * G.P.SR)));
        return [s0, s1, s0 + G.bestWindowStart(long.subarray(s0, s1))]; }});
      process.stdout.write(JSON.stringify({{ out, bounds }}));
    """
    js = json.loads(run_node(script))
    from feature_extraction.gtm_spectrogram import segment_bounds, segment_window

    for y, r in zip(cases, js["out"]):
        ws = best_window_start(y)
        w = fit_input(y[ws: ws + GTM_INPUT_SAMPLES])
        assert ws == r["ws"] and w.size == r["len"] == GTM_INPUT_SAMPLES
        assert int(np.argmax(w != 0)) == r["first"]
    long = cases[-1]
    for (a, b), (s0, s1, ws) in zip(segments, js["bounds"]):
        assert segment_bounds(len(long), a, b) == (s0, s1)
        assert segment_window(long, a, b)[1] == ws


@needs_gtm_model
def test_exported_gtm_model_labels_and_shape():
    from gtm_model.inference import GTMClassifier

    clf = GTMClassifier()
    assert clf.labels == class_names()
    assert clf.model.input_shape == (GTM_FRAMES, GTM_BINS, 1)
    x = normalize(gtm_spectrogram(tone(1.1, 900, 0.5, sr=44100)))
    scores = clf.predict_inputs([x])[0]
    assert set(scores) == set(class_names()) and abs(sum(scores.values()) - 1) < 1e-5


@needs_node
@needs_gtm_model
def test_browser_gtm_path_matches_python_pipeline(tmp_path):
    """End to end: static/js/gtm.js + the app's TensorFlow.js build, run in Node on the exported
    model files, give the same windows and probabilities as the Python pipeline used for training."""
    from gtm_model import pipeline
    from gtm_model.inference import GTMClassifier

    sr = 22050
    rng = np.random.default_rng(5)
    y = (0.02 * rng.standard_normal(int(4.3 * sr))).astype(np.float32)
    y[int(0.6 * sr):int(1.1 * sr)] += tone(0.5, 950, 0.6, sr)
    burst = int(0.12 * sr)
    y[int(2.7 * sr):int(2.7 * sr) + burst] += (rng.standard_normal(burst) * np.exp(-np.arange(burst) / (0.02 * sr))).astype(np.float32)
    audio = pipeline.upload_audio(np.clip(y, -1, 1), sr)
    segs = [s for s, _ in pipeline.app_segments(audio)]
    assert segs
    audio.y44.astype("<f4").tofile(tmp_path / "y44.f32")
    jobs = [{"start": round(s.start, 3), "end": round(s.end, 3)} for s in segs]
    folder = GTM_MODEL_DIR.as_posix()
    script = f"""
      const fs = require('fs'); const path = require('path');
      globalThis.window = {{ tf: require({json.dumps(TF_JS.as_posix())}) }};
      globalThis.fetch = async (url) => {{
        const file = path.join({json.dumps(folder)}, new URL(url).pathname.slice(1));
        return new Response(fs.readFileSync(file), {{ status: 200 }});
      }};
      require({json.dumps(GTM_JS.as_posix())});
      (async () => {{
        await window.tf.setBackend('cpu');
        const G = window.SSGTM;
        const model = new G.GTMModel({{ enabled: true, model_url: 'http://gtm.local/model.json', metadata_url: 'http://gtm.local/metadata.json' }});
        await model.load();
        const buf = fs.readFileSync({json.dumps((tmp_path / 'y44.f32').as_posix())});
        const y44 = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
        const out = [];
        for (const s of {json.dumps(jobs)}) out.push(await model.classifySegment(y44, s.start, s.end));
        process.stdout.write(JSON.stringify({{ labels: model.labels, out }}));
      }})().catch(e => {{ console.error(e); process.exit(1); }});
    """
    js = json.loads(run_node(script))
    clf = GTMClassifier()
    assert js["labels"] == clf.labels
    for seg, res in zip(segs, js["out"]):
        x, window_start = pipeline.segment_input(audio, seg)
        py = clf.predict_inputs([x])[0]
        assert abs(res["window_start"] - window_start) < 1e-9
        assert max(abs(res["scores"][c] - py[c]) for c in clf.labels) < 1e-4


def test_label_check_and_score_parsing(app):
    from src.services import gtm_service

    with app.app_context():
        labels = class_names()
        assert gtm_service.label_check(labels)["ok"]
        bad = gtm_service.label_check([*labels[:-1], "Background noise"])
        assert not bad["ok"] and "Background Noise" in bad["missing"]
        scores = {c: 1 / len(labels) for c in labels}
        parsed = gtm_service.parse_scores(scores, labels)
        assert abs(sum(parsed.values()) - 1) < 1e-9
        with pytest.raises(ValueError):
            gtm_service.parse_scores({c: 0.5 for c in labels}, labels)  # does not sum to 1
        with pytest.raises(ValueError):
            gtm_service.parse_scores({labels[0]: 1.0}, labels)  # missing classes
        with pytest.raises(ValueError):
            gtm_service.parse_scores({**scores, labels[0]: float("nan")}, labels)
        assert gtm_service.clean_version("<script>tm@2026") == "scripttm@2026"


def _enable_fixture(app, folder: Path, monkeypatch):
    import config
    from src.services import gtm_service

    monkeypatch.setattr(config, "GTM_MODEL_DIR", folder)
    monkeypatch.setattr(gtm_service, "GTM_MODEL_DIR", folder)
    import src.blueprints.gtm as gtm_bp

    monkeypatch.setattr(gtm_bp, "GTM_MODEL_DIR", folder)


@pytest.mark.skipif(not (Path(__file__).parent / "fixtures" / "gtm_test_model" / "model.json").exists(), reason="fixture missing")
def test_gtm_files_served_and_results_finalise_record(app, client, gtm_fixture_dir, monkeypatch, tmp_path):
    from .conftest import model_available, write

    if not model_available():
        pytest.skip("no trained Python model")
    _enable_fixture(app, gtm_fixture_dir, monkeypatch)
    login(client, "admin")
    assert client.get("/gtm/model/model.json").status_code == 200
    assert client.get("/gtm/model/../config.py").status_code == 404
    path = write(tmp_path / "siren.wav", tone(3.0, 900, 0.5))
    res = upload(client, path).get_json()
    assert res["ok"] and res["record"]["gtm_status"] == "pending"
    job = res["gtm_job"]
    assert "python" not in json.dumps(job).lower() or "python" not in job  # job carries no Python results
    labels = class_names()
    other = round(0.09 / max(1, len(labels) - 1), 6)
    siren = round(1.0 - (len(labels) - 1) * other, 6)
    segs = [{"index": s["index"], "scores": {c: (siren if c == "Alarm or Siren" else other) for c in labels}}
            for s in job["segments"]]
    r = client.post(job["submit_url"], json={"model_version": "fixture@1", "labels": labels, "segments": segs})
    body = r.get_json()
    assert r.status_code == 200 and body["record"]["gtm_status"] == "completed"
    assert body["record"]["gtm"]["top"] == "Alarm or Siren"
    # A second submission is refused (results are final).
    assert client.post(job["submit_url"], json={"model_version": "x", "labels": labels, "segments": segs}).status_code == 400


def test_gtm_error_payload_falls_back_to_single_model(app, client, gtm_fixture_dir, monkeypatch, tmp_path):
    from .conftest import model_available, write

    if not model_available():
        pytest.skip("no trained Python model")
    _enable_fixture(app, gtm_fixture_dir, monkeypatch)
    login(client, "admin")
    res = upload(client, write(tmp_path / "x.wav", tone(2.0))).get_json()
    r = client.post(res["gtm_job"]["submit_url"], json={"error": "model failed to load"})
    rec = r.get_json()["record"]
    assert rec["gtm_status"] == "failed" and rec["consistency"] == "Single-Model Result"
    with app.app_context():
        from src.models import SystemAlert

        assert SystemAlert.query.filter_by(kind="model_failure").count() == 1


def test_gtm_label_mismatch_is_rejected(app, client, gtm_fixture_dir, monkeypatch, tmp_path):
    from .conftest import model_available, write

    if not model_available():
        pytest.skip("no trained Python model")
    _enable_fixture(app, gtm_fixture_dir, monkeypatch)
    login(client, "admin")
    res = upload(client, write(tmp_path / "x.wav", tone(2.0))).get_json()
    wrong = [c.lower() for c in class_names()]
    segs = [{"index": s["index"], "scores": {c: 0.1 for c in wrong}} for s in res["gtm_job"]["segments"]]
    r = client.post(res["gtm_job"]["submit_url"], json={"model_version": "x", "labels": wrong, "segments": segs})
    assert r.get_json()["record"]["gtm_status"] == "failed"
