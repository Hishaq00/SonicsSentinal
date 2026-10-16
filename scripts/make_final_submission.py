"""Lay out the final submission: one numbered folder per checklist item.

The checklist has twenty items. Sixteen of them can be filled from this
repository; four need something only the team can supply (the GitHub URL, the
deployment URL, the demonstration video, and confirmation of the team record).
Those four get a short brief instead of an invented answer, so nothing in the
bundle is a placeholder pretending to be finished work.

    python scripts/make_final_submission.py
    python scripts/make_final_submission.py --no-datasets   # skip the 4.5 GB copy
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_submission import ROOT, active_model, tracked_files  # noqa: E402

DOCS = ROOT / "documentation"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"), encoding="utf-8", newline="\r\n")


def copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"    missing, skipped: {src.relative_to(ROOT)}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree(src: Path, dst: Path, pattern: str = "*") -> int:
    if not src.is_dir():
        print(f"    missing, skipped: {src}")
        return 0
    n = 0
    for p in sorted(src.rglob(pattern)):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
            continue
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        n += 1
    return n


def build_source_code(dst: Path) -> int:
    """Item 03: the public-repository contents, exactly as make_submission.py picks them."""
    keep = active_model()
    n = 0
    for rel in tracked_files():
        src = ROOT / rel
        if not src.is_file():
            continue
        parts = Path(rel).parts
        superseded = (
            len(parts) == 3 and parts[0] == "python_models" and parts[1] == "saved"
            and rel.endswith(".joblib") and parts[2] != keep
        )
        if superseded:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        n += 1
    for d in ("data/storage", "database"):
        (dst / d).mkdir(parents=True, exist_ok=True)
        (dst / d / ".gitkeep").touch()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT.parent.parent / "submission" / "Final Submission"))
    ap.add_argument("--no-datasets", action="store_true",
                    help="leave items 04-06 empty (the audio splits are 4.5 GB)")
    ap.add_argument("--docs-only", action="store_true",
                    help="rebuild every item except 04-06, keeping datasets already copied")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    keep_dirs = {"04 - Training Dataset", "05 - Validation Dataset", "06 - Testing Dataset"}
    if out.exists():
        for child in out.iterdir():
            if args.docs_only and child.name in keep_dirs:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    out.mkdir(parents=True, exist_ok=True)

    F = {
        1: out / "01 - Project Report",
        2: out / "02 - Public GitHub URL",
        3: out / "03 - Complete Source Code",
        4: out / "04 - Training Dataset",
        5: out / "05 - Validation Dataset",
        6: out / "06 - Testing Dataset",
        7: out / "07 - Dataset Metadata",
        8: out / "08 - Python Model",
        9: out / "09 - GTM Model",
        10: out / "10 - Preprocessing Scripts",
        11: out / "11 - Feature Extraction Scripts",
        12: out / "12 - Alert Rules",
        13: out / "13 - Model Comparison Report",
        14: out / "14 - Installation Instructions",
        15: out / "15 - Execution Instructions",
        16: out / "16 - Deployment URL",
        17: out / "17 - Demonstration Video",
        18: out / "18 - Technical Blog",
        19: out / "19 - AI Usage",
        20: out / "20 - Team Contribution Record",
    }
    for d in F.values():
        d.mkdir(parents=True, exist_ok=True)

    print("  01 Project report")
    for name in ("PROJECT_REPORT.md", "FUNCTIONAL_REQUIREMENTS_AUDIT_REPORT.md", "DEVELOPMENT_LOG.md"):
        copy(DOCS / name, F[1] / name)

    print("  02 Public GitHub URL        [needs your input]")
    write(F[2] / "GITHUB_URL.md", GITHUB_URL_MD)

    print("  03 Complete source code")
    n = build_source_code(F[3])
    print(f"     {n} files")

    if args.docs_only:
        print("  04-06 Datasets             [kept: --docs-only]")
    elif args.no_datasets:
        print("  04-06 Datasets             [skipped: --no-datasets]")
        for i, split in ((4, "train"), (5, "validation"), (6, "test")):
            write(F[i] / "README.md", dataset_readme(split, None, None))
    else:
        for i, split in ((4, "train"), (5, "validation"), (6, "test")):
            src = ROOT / "audio_dataset" / split
            print(f"  {i:02d} {split} dataset - copying, this takes a while...")
            count = copy_tree(src, F[i] / split)
            size = sum(p.stat().st_size for p in (F[i] / split).rglob("*") if p.is_file())
            print(f"     {count} files, {size / 1e9:.2f} GB")
            write(F[i] / "README.md", dataset_readme(split, count, size))

    print("  07 Dataset metadata")
    copy(ROOT / "data" / "dataset_metadata.csv", F[7] / "dataset_metadata.csv")
    copy(ROOT / "data" / "dataset_statistics.json", F[7] / "dataset_statistics.json")
    copy(ROOT / "audio_dataset" / "metadata.csv", F[7] / "audio_dataset_metadata.csv")
    copy(ROOT / "data" / "features" / "feature_config.json", F[7] / "feature_config.json")
    copy(ROOT / "reports" / "dataset_report.md", F[7] / "dataset_report.md")
    copy(ROOT / "audio_dataset" / "README.md", F[7] / "audio_dataset_README.md")

    print("  08 Python model")
    keep = active_model()
    saved = ROOT / "python_models" / "saved"
    if keep:
        copy(saved / keep, F[8] / keep)
        copy(saved / f"{Path(keep).stem}.json", F[8] / f"{Path(keep).stem}.json")
    copy(saved / "active.json", F[8] / "active.json")
    copy_tree(ROOT / "python_models", F[8] / "training_scripts", "*.py")
    write(F[8] / "README.md", python_model_md(keep))

    print("  09 GTM model")
    copy_tree(ROOT / "gtm_model" / "model", F[9] / "model")
    copy_tree(ROOT / "gtm_model", F[9] / "integration", "*.py")
    copy(ROOT / "config" / "classes.json", F[9] / "classes.json")
    write(F[9] / "README.md", GTM_MD)

    print("  10 Preprocessing scripts")
    copy_tree(ROOT / "audio_preprocessing", F[10] / "audio_preprocessing")
    copy_tree(ROOT / "augmentation", F[10] / "augmentation")
    write(F[10] / "README.md", PREPROC_MD)

    print("  11 Feature-extraction scripts")
    copy_tree(ROOT / "feature_extraction", F[11] / "feature_extraction")
    copy(ROOT / "data" / "features" / "feature_config.json", F[11] / "feature_config.json")
    write(F[11] / "README.md", FEATURES_MD)

    print("  12 Alert rules")
    copy_tree(ROOT / "alert_rules", F[12] / "alert_rules")
    write(F[12] / "README.md", ALERTS_MD)

    print("  13 Model-comparison report")
    for name in ("model_prediction_and_confidence_comparison_summary.md",
                 "model_prediction_and_confidence_comparison_report.csv"):
        copy(ROOT / "reports" / name, F[13] / name)
    copy_tree(ROOT / "reports" / "python_model", F[13] / "supporting" / "python_model")
    copy_tree(ROOT / "reports" / "gtm_model", F[13] / "supporting" / "gtm_model")

    print("  14 Installation instructions")
    write(F[14] / "INSTALLATION.md", INSTALL_MD)
    copy(ROOT / "requirements.txt", F[14] / "requirements.txt")
    copy(ROOT / "requirements-gtm.txt", F[14] / "requirements-gtm.txt")
    copy(ROOT / ".env.example", F[14] / "env.example.txt")

    print("  15 Execution instructions")
    write(F[15] / "EXECUTION.md", EXEC_MD)
    copy(ROOT / "HOW_TO_RUN.txt", F[15] / "HOW_TO_RUN.txt")

    print("  16 Deployment URL           [needs your input]")
    write(F[16] / "DEPLOYMENT_URL.md", DEPLOY_MD)

    print("  17 Demonstration video      [needs your input]")
    write(F[17] / "RECORDING_BRIEF.md", VIDEO_MD)

    print("  18 Technical blog")
    copy(DOCS / "TECHNICAL_BLOG.md", F[18] / "TECHNICAL_BLOG.md")

    print("  19 AI usage")
    copy(ROOT / "AI_USAGE.md", F[19] / "AI_USAGE.md")

    print("  20 Team contribution record [needs your input: names are placeholders]")
    copy(DOCS / "TEAM_CONTRIBUTIONS.md", F[20] / "TEAM_CONTRIBUTIONS.md")

    write(out / "00 - START HERE.md", START_MD)
    copy(DOCS / "SUBMISSION_CHECKLIST.md", out / "00 - Original checklist (project copy).md")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"\n  done: {files} files, {total / 1e9:.2f} GB -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# Authored files
# --------------------------------------------------------------------------- #
START_MD = """
# Final submission - what is here

One folder per checklist item. Sixteen are filled from the repository; four need
something the team has to supply, and each of those holds a short brief saying
exactly what and where.

| # | Item | Status |
|---|------|--------|
| 01 | Project report | ready - `PROJECT_REPORT.md` (+ requirements audit, development log) |
| 02 | Public GitHub URL | **you** - repository not pushed yet, see `GITHUB_URL.md` |
| 03 | Complete source code | ready - the public-repository contents |
| 04 | Training dataset | ready - `train/` |
| 05 | Validation dataset | ready - `validation/` |
| 06 | Testing dataset | ready - `test/` |
| 07 | Dataset metadata | ready - metadata CSVs, statistics, dataset report |
| 08 | Python model | ready - the active saved model + training scripts |
| 09 | GTM model | ready - exported model, integration code, label file |
| 10 | Preprocessing scripts | ready |
| 11 | Feature-extraction scripts | ready |
| 12 | Alert rules | ready |
| 13 | Model-comparison report | ready - summary + 170-row CSV + confusion matrices |
| 14 | Installation instructions | ready |
| 15 | Execution instructions | ready |
| 16 | Deployment URL | **you** - not deployed yet, see `DEPLOYMENT_URL.md` |
| 17 | Demonstration video | **you** - not recorded, see `RECORDING_BRIEF.md` |
| 18 | Technical blog | ready |
| 19 | AI_USAGE.md | ready |
| 20 | Team contribution record | **you** - member names are still `[name]` placeholders |

## Before you submit

1. **Push the repository** and put the URL in `02`. The saved model is 155 MB, and
   GitHub rejects any file over 100 MB - `02/GITHUB_URL.md` covers the two ways round it.
2. **Deploy** and put the live URL in `16`. The microphone needs HTTPS, so a plain
   `http://` address will leave live monitoring dead.
3. **Record the video** using the shot list in `17`.
4. **Fill in `20/TEAM_CONTRIBUTIONS.md`** - every row still reads `_[name]_`. The brief
   requires each member to own named modules, and the file suggests a split to start from.
5. **Consider a UI section in `01/PROJECT_REPORT.md`** - the report covers the data model,
   the requirements traceability and the dataset, but has no interface chapter, so the
   dashboard and the live monitor screens are currently undocumented.

Rebuild this folder any time with `python scripts/make_final_submission.py`.
"""

GITHUB_URL_MD = """
# Public GitHub URL

**Status: not pushed yet.** Put the URL here once the repository is public:

    https://github.com/<user>/<repo>

## Pushing it

The contents of `03 - Complete Source Code` are what belongs in the repository -
`.gitignore` is already included, so the recorded audio and the feature caches
stay out.

    cd "03 - Complete Source Code"
    git init -b main
    git add -A
    git commit -m "SonicSentinel AI"
    git remote add origin https://github.com/<user>/<repo>.git
    git push -u origin main

## One thing will break that push

`python_models/saved/py-v12-ensemble-weighted.joblib` is **155 MB**, and GitHub
refuses any single file over 100 MB. Pick one:

**A - Git LFS** (keeps the best model, 88.5% CV macro-F1)

    git lfs install
    git lfs track "*.joblib"
    git add .gitattributes
    # then commit and push as above

**B - ship a smaller model instead** (plain git, no LFS)

Swap the active model to `py-v3-ensemble-weighted` (5.3 MB) by editing
`python_models/saved/active.json`, and copy that `.joblib` in. Costs accuracy:
80.2% instead of 88.5% CV macro-F1.

`data/features/features.npz` is 63.6 MB - over GitHub's 50 MB warning threshold
but under the hard limit, so it pushes either way.
"""

DEPLOY_MD = """
# Deployment URL

**Status: not deployed yet.** Put the live URL here:

    https://<your-domain>

## The microphone needs HTTPS

`getUserMedia()` only works on a secure origin (or localhost). Over plain
`http://` the browser blocks the microphone outright, so live monitoring and the
dashboard waveforms will not run. Terminate TLS before you demonstrate.

## Deploying on a VPS

    # 1. upload and unpack
    scp sonicsentinel-deploy.zip user@server:/tmp/
    sudo mkdir -p /opt/sonicsentinel && sudo chown $USER /opt/sonicsentinel
    unzip /tmp/sonicsentinel-deploy.zip -d /opt/sonicsentinel && cd /opt/sonicsentinel

    # 2. system packages (FFmpeg is bundled by imageio-ffmpeg, so it is not needed)
    sudo apt update && sudo apt install -y python3.12 python3.12-venv libsndfile1 nginx

    # 3. dependencies
    python3.12 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt

    # 4. configuration
    cp .env.example .env
    python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(48))"
    #   paste that into .env, and set SESSION_COOKIE_SECURE=1

    # 5. first start creates the database and the demo accounts
    python -c "from src import create_app; create_app()"

Then run it under systemd behind nginx, and get a certificate:

    ExecStart=/opt/sonicsentinel/.venv/bin/gunicorn --workers 2 --threads 4 \\
              --timeout 120 --bind 127.0.0.1:8000 wsgi:app

    sudo certbot --nginx -d <your-domain>

Set `client_max_body_size 30M;` in nginx to match `MAX_UPLOAD_MB`.

**Memory:** the saved ensemble is 155 MB and `WARMUP_MODELS=1` loads it at start,
once per worker. Two workers need about 2 GB of RAM; on a 1 GB box use
`--workers 1` or set `WARMUP_MODELS=0`.

The deployment archive is built by `python scripts/make_deploy_zip.py`.
"""

VIDEO_MD = """
# Demonstration video

**Status: not recorded.** Put the file or the link here when it is done.

Record against the deployed HTTPS site, not `http://` - the microphone will not
start otherwise.

## Shot list

1. **Landing page** - scroll through it; the hero, the pipeline, the two-model comparison.
2. **Sign in** as `admin` (`Admin@2026`).
3. **Dashboard** - the instrument header, the figures, then press *Listen live*:
   make a noise and show the waveform reacting and the panel turning red on a
   detected sound.
4. **Analyze audio** - upload a clip from `sample_audio/`. Show the Python result
   arriving first, the GTM model then running in the browser, and the combined
   decision with its confidence bars.
5. **Live monitor** - start monitoring, make a sharp sound (a clap, keys, an alarm
   on a phone). Show the waveform, the scrolling spectrogram, the 3D spectrum, the
   per-window table, and an alert being raised.
6. **Alerts** - acknowledge one, then resolve it.
7. **Review queue** - open an item where the two models disagreed and correct it.
8. **Admin > Models** - both models side by side and the label-match check.
9. **Admin > Model comparison** - the 170-recording evaluation.
10. Close on the privacy notice, to show the ethical framing.

Keep each step short; the interesting part is the two models disagreeing and the
alert rules deciding what to do about it.
"""


def dataset_readme(split: str, count: int | None, size: int | None) -> str:
    if count is None:
        body = ("This folder is empty: the build was run with `--no-datasets`.\n\n"
                f"Copy `audio_dataset/{split}/` from the project into this folder.")
    else:
        body = f"`{split}/` holds **{count} audio files**, {size / 1e9:.2f} GB.\n"
    return f"""
# {split.capitalize()} dataset

{body}

The split is the one the models were trained and evaluated on. `07 - Dataset
Metadata` carries the per-file metadata (class, split, duration, source) and the
class statistics.

The splits are stratified per class and disjoint: no recording appears in more
than one of train / validation / test, so the test figures in the
model-comparison report are on unseen audio.
"""


def python_model_md(keep: str | None) -> str:
    return f"""
# Python model

`{keep or "(no active model)"}` is the saved model the application loads, named by
`active.json`. Its `.json` card next to it records the algorithm, the class list,
the hyperparameters, the feature configuration and the metrics.

* **Algorithm:** weighted ensemble over three members, combined as a weighted
  mean of class probabilities
* **Features:** 193 per 2-second segment, feature set `ssf-1.0` (see
  `11 - Feature Extraction Scripts`)
* **Classes:** 17 - the 10 mandatory categories plus 7 optional ones
* **Cross-validated macro F1:** 88.5%
* **Test accuracy in the 170-recording comparison:** 91.4% (see `13`)

`training_scripts/` holds the code that produced it - `train.py` for this model,
`train_cnn.py` for the CNN variant, `inference.py` for the loading and prediction
path the web application calls, and `evaluation.py` and `ensemble.py` for the rest.

Retrain with:

    python -m feature_extraction.build_features
    python -m python_models.train

Older versions are not included here; the brief asks for the saved model, and the
eleven superseded binaries add 130 MB. Their metric cards are still in the
repository under `python_models/saved/`.
"""


GTM_MD = """
# GTM model

`model/` is the Google Teachable Machine audio model exported for TensorFlow.js:

* `model.json` - the graph
* `weights.bin` - the weights, 5.85 MB
* `metadata.json` - the labels and the exporter's settings

It runs **in the browser**, not on the server: the live monitor and the analysis
page load it with TensorFlow.js and classify the same 2-second segments the Python
model sees, so the two answers can be compared per window. The application serves
these three files from `/gtm/model/<file>`.

`integration/` holds the server-side Python that wraps it - `pipeline.py`,
`inference.py` and `export_gtm_samples.py`.

`classes.json` is the label file both models share. Admin > Models runs a check
that the GTM labels and the Python model's classes still match; a mismatch there
is what stops the two results from being comparable.
"""

PREPROC_MD = """
# Preprocessing scripts

`audio_preprocessing/` is what every recording passes through before it reaches
either model, whether it arrives as an upload or as a window off the microphone:

* `loader.py` - decoding, resampling to 22.05 kHz, mono conversion
* `validation.py` - format, duration and size checks on what was submitted
* `preprocess.py` - trimming, normalising and segmenting into 2-second windows
* `noise_reduction.py` - the noise-floor estimate and its subtraction
* `quality.py` - the Good / Acceptable / Poor / Unusable label shown against each
  event, from the clipping, silence and signal-to-noise tests
* `fingerprint.py` - the content hash used to recognise a re-upload of the same audio

`augmentation/` is training-time only - time shift, noise, pitch and gain
variants used to balance the smaller classes. `transforms.py` holds the
individual operations, `augment_dataset.py` applies them across a split.

The same preprocessing runs at training time and at inference time; that is
deliberate, because a mismatch between the two is the usual reason a model that
scores well offline behaves differently live.
"""

FEATURES_MD = """
# Feature-extraction scripts

`feature_extraction/` turns a 2-second segment into the 193 numbers the Python
model reads:

* `build_features.py` - walks the dataset and writes `data/features/features.npz`
  plus `feature_config.json`
* `gtm_spectrogram.py` - the mel spectrogram rendering, kept in step with what the
  GTM model expects so both models see comparable input

`feature_config.json` records the exact settings: 22.05 kHz, 2048-sample FFT,
512 hop, 20 MFCCs, 64 mel bands summarised into 32, 20 Hz floor, f0 tracked
between 60 and 2000 Hz. The feature set is versioned (`ssf-1.0`) and each saved
model stores the configuration it was trained with, so a model is never fed
features built with different settings.

Rebuild with:

    python -m feature_extraction.build_features
"""

ALERTS_MD = """
# Alert rules

`alert_rules/` decides whether a classification becomes an alert, and who is
responsible for it:

* `alert_rules.json` - the rules themselves: per class, a severity, a minimum
  confidence, how many consecutive detections are required, whether the two
  models must agree, the recommended action and the role it is routed to
* `rules.py` - loading and validation
* `engine.py` - evaluation, the consecutive-detection tracking, escalation and
  the de-duplication that stops one long sound raising twenty alerts

A rule needing model agreement is what keeps a single confident-but-wrong model
from paging the security desk; a rule needing consecutive detections is what
keeps a one-off transient from doing it. Administrators can edit all of this at
Admin > Alert rules without touching the file by hand.
"""

INSTALL_MD = """
# Installation

## What you need

* **Python 3.12-3.14** - tested with 3.14.7 locally and 3.12.7 on the deployment platform
* **Chrome, Edge or Firefox** - the GTM model runs inside the browser
* A microphone, for live monitoring
* Optional: Node.js, used only by some GTM comparison tests (skipped without it)

FFmpeg is **not** required: `imageio-ffmpeg` brings its own binary for M4A/AAC
decoding. On Linux install `libsndfile1` if the wheel does not cover your platform.

## Setup

Open a terminal in the source folder - the one containing `run.py`.

**Windows**

    python -m venv .venv
    .venv\\Scripts\\python.exe -m pip install -r requirements.txt
    copy .env.example .env

**Linux / macOS**

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    cp .env.example .env

Then open `.env` and set `SECRET_KEY` to a long random string:

    python -c "import secrets; print(secrets.token_urlsafe(48))"

Everything else in `.env` can stay as it is for a local run. `SEED_DEMO_USERS=1`
creates the demo accounts on first start.

## Optional: GTM retraining dependencies

Only needed if you intend to retrain the browser model - it installs PyTorch:

    .venv\\Scripts\\python.exe -m pip install -r requirements-gtm.txt

## What the first start creates

* `database/sonicsentinel.db` - the SQLite database, from the schema in `database/schema.sql`
* the demo accounts
* `data/storage/` - uploaded audio and generated spectrogram images

Nothing else needs to be prepared: the saved Python model and the exported GTM
model are both included.
"""

EXEC_MD = """
# Execution

## Start it

    Windows:        .venv\\Scripts\\python.exe run.py
    Linux / macOS:  .venv/bin/python run.py

Wait for:

    SonicSentinel AI running on http://127.0.0.1:5000  (Ctrl+C to stop)

Open **http://127.0.0.1:5000**. Use `127.0.0.1`, not the machine's LAN address -
browsers only allow the microphone on localhost or over HTTPS. Change the port
with `PORT=5001` in `.env`.

## Accounts

| Username | Password | Role |
|---|---|---|
| admin | Admin@2026 | Administrator - everything |
| evaluator | Evaluate@2026 | Evaluation runs |
| reviewer | Review@2026 | Manual review queue |
| security | Secure@2026 | Live monitoring and alerts |
| maintenance | Maintain@2026 | Machinery fault alerts |
| user | User@2026 | Normal user - own uploads only |

## Walking through it

**Analyze a file** - *Analyze audio* (`/analyze`), drop in a clip from
`sample_audio/`, press Analyze. The Python model answers from the server first;
the GTM model then runs in the browser on the same segments and the final decision
combines the two. Keep the tab open until the GTM result arrives.

**Live microphone** - *Live monitor* (`/live`), Start monitoring, allow the
microphone. Each window is classified by both models, compared, and checked
against the alert rules. The waveform, the scrolling spectrogram and the 3D
spectrum follow the input.

**Dashboard** - press *Listen live* in the header to see the waveform react to the
room without starting a full monitoring session.

**Results** - Timeline, Event history, Alerts, Review queue. Admin > Models shows
both models and verifies the GTM labels still match the Python classes.

## Tests

    .venv\\Scripts\\python.exe -m pytest tests -q
    .venv\\Scripts\\python.exe -m pytest tests/test_gtm.py -v

## Retraining

    python -m feature_extraction.build_features
    python -m python_models.train

    python -m pip install -r requirements-gtm.txt
    python -m scripts.train_gtm_model            # 30-60 min on a laptop CPU
    python -m scripts.train_gtm_model --resume   # continue after an interruption

    python -m scripts.generate_100_comparison_report

The application picks up new model files by itself - reload the page, no restart.

## Production

    Windows:  .venv\\Scripts\\waitress-serve.exe --host 0.0.0.0 --port 8000 wsgi:app
    Linux:    .venv/bin/gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:8000 wsgi:app
    Docker:   docker build -t sonicsentinel . && docker run -p 8000:8000 -e SECRET_KEY=... sonicsentinel

Serve over HTTPS and set `SESSION_COOKIE_SECURE=1`. See `16 - Deployment URL`.

## If something goes wrong

* **Address already in use** - change `PORT` in `.env`.
* **GTM result says unavailable** - Admin > Models: GTM must be enabled, the files
  present in `gtm_model/model/`, and the label check must pass.
* **Microphone will not start** - open the site as `http://127.0.0.1:5000` or over HTTPS.
"""

if __name__ == "__main__":
    sys.exit(main())
