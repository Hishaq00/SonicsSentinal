# Installation

## 1. Prerequisite software

| Requirement | Detail |
|---|---|
| **Python** | **3.12 – 3.14** — tested with 3.14.7 locally and 3.12.7 on the deployment platform |
| **Browser** | **Chrome, Edge or Firefox** — the GTM model runs inside the browser via TensorFlow.js |
| **Microphone** | required only for live monitoring |
| **Node.js** | *optional* — used by a few GTM comparison tests, which skip themselves if it is absent |
| **Git** | optional, for cloning instead of copying the folder |

## 2. Supported operating systems

| OS | Status |
|---|---|
| Windows 10 / 11 | tested (development machine: Windows 11) |
| Linux (Debian/Ubuntu) | tested (deployment platform) |
| macOS 12+ | supported — same commands as Linux |

## 3. FFmpeg

**FFmpeg does not need to be installed separately.** `imageio-ffmpeg`, pulled in by `requirements.txt`,
ships its own binary and that is what decodes M4A/AAC uploads.

On Linux, if the `soundfile` wheel does not cover your platform, install the system audio library:

    sudo apt-get install libsndfile1

## 4. Audio-library configuration

No manual configuration is needed. The audio stack is installed by `requirements.txt`:

| Library | Used for |
|---|---|
| `librosa` | resampling, mel spectrograms, MFCC and the rest of the 193 features |
| `soundfile` | WAV / FLAC / OGG decoding |
| `imageio-ffmpeg` | MP3 / M4A / AAC decoding (bundled FFmpeg binary) |
| `numpy`, `scipy` | signal processing |

Everything is resampled internally to **22.05 kHz mono** for the Python model and **44.1 kHz** for the GTM
spectrogram path. Those rates are fixed in `feature_config.json` and should not be changed by hand — a model
is never fed features built with different settings.

## 5. Virtual environment and dependencies

Open a terminal in the source folder — the one containing `run.py`.

**Windows**

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    copy .env.example .env

**Linux / macOS**

    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    cp .env.example .env

### Optional: GTM retraining dependencies

Only needed if you intend to retrain the browser model — it installs PyTorch:

    .venv\Scripts\python.exe -m pip install -r requirements-gtm.txt

## 6. Environment variables

Open `.env` and set `SECRET_KEY` to a long random string:

    python -c "import secrets; print(secrets.token_urlsafe(48))"

Everything else can stay at its default for a local run.

| Variable | Default | Meaning |
|---|---|---|
| `SECRET_KEY` | *(must be set)* | signs sessions — required in production |
| `DATABASE_URL` | SQLite at `database/sonicsentinel.db` | set to a `postgresql+psycopg://…` URL to use PostgreSQL |
| `STORAGE_DIR` | `data/storage` | where uploaded audio and generated images go |
| `MAX_UPLOAD_MB` | `30` | per-request upload limit; the admin setting must be ≤ this |
| `SESSION_COOKIE_SECURE` | `0` | set to `1` when serving over HTTPS |
| `SESSION_HOURS` | `8` | session lifetime |
| `SEED_DEMO_USERS` | `1` | create the demo / evaluator accounts on first start |
| `WARMUP_MODELS` | `1` | pre-load the Python model at start-up so the first prediction is fast |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | development server binding |
| `FLASK_DEBUG` | `0` | Flask debug mode |

## 7. Database configuration and initialization

**Default — SQLite, no setup required.** On first start the application creates
`database/sonicsentinel.db` from the schema in `database/schema.sql`. There is no separate init command to
run.

**PostgreSQL (optional).** Create the database, then point `DATABASE_URL` at it:

    DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/sonicsentinel

The schema is applied on first start in the same way.

To start over from an empty database, stop the app and delete `database/sonicsentinel.db`; it is recreated
on the next start.

## 8. Folder setup

Nothing needs to be created by hand. The first start creates:

| Path | Contents |
|---|---|
| `database/sonicsentinel.db` | the SQLite database |
| `data/storage/` | uploaded audio and generated spectrogram images |

## 9. Python model placement

**The trained model is already included — no placement step is required.**

| Path | Contents |
|---|---|
| `python_models/saved/py-v12-ensemble-weighted.joblib` | the model the application loads |
| `python_models/saved/py-v12-ensemble-weighted.json` | its model card — algorithm, classes, hyperparameters, feature config, metrics |
| `python_models/saved/active.json` | names which saved model is active |

To install a different model, copy its `.joblib` and `.json` pair into `python_models/saved/` and point
`active.json` at it, or switch it from **Admin → Models**. The application picks up a new model file by
itself — reload the page, no restart needed.

A copy of the active model and its training scripts is also in `08 - Python Model/`.

## 10. GTM model placement / URL configuration

**The exported GTM model is already included — no placement step is required.**
It lives in `gtm_model/model/` as `model.json`, `weights.bin` and `metadata.json`, and the application
serves it from `/gtm/model/<file>`.

**Admin → Models → Teachable Machine** chooses the source:

* **Local files** (default) — the app serves `gtm_model/model/`. To replace the model, copy the three
  exported files into that folder.
* **Hosted link** — paste a Teachable Machine share link:

      https://teachablemachine.withgoogle.com/models/<id>/

Either way the model's labels must match `config/classes.json` exactly, or the application rejects it. The
choice is saved in the settings and written to the audit trail.

A copy of the exported model is also in `09 - GTM Model/model/`.

## 11. Execution command

**Windows**

    .venv\Scripts\python.exe run.py

**Linux / macOS**

    .venv/bin/python run.py

Wait for:

    SonicSentinel AI running on http://127.0.0.1:5000  (Ctrl+C to stop)

Then open **http://127.0.0.1:5000**. Use `127.0.0.1`, not the machine's LAN address — browsers only allow
microphone access on localhost or over HTTPS.

Full walkthrough: `15 - Execution Instructions/EXECUTION.md`.

## 12. Test command

    .venv\Scripts\python.exe -m pytest tests -q

Individual suites:

    .venv\Scripts\python.exe -m pytest tests/test_gtm.py -v
    .venv\Scripts\python.exe -m pytest tests/test_alert_engine.py -v
    .venv\Scripts\python.exe -m pytest tests/test_security_db_live.py -v

On Linux / macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

## 13. Administrator setup

With `SEED_DEMO_USERS=1` (the default) the first start creates these accounts:

| Username | Password | Role |
|---|---|---|
| `admin` | `Admin@2026` | Administrator — full access |
| `evaluator` | `Evaluate@2026` | Competition Evaluator |
| `reviewer` | `Review@2026` | Audio Reviewer |
| `security` | `Secure@2026` | Security Operator |
| `maintenance` | `Maintain@2026` | Maintenance Personnel |
| `user` | `User@2026` | Normal User |

**Change `admin`'s password before exposing the site publicly.** Sign in as `admin`, then
**Admin → Users & roles** to change passwords, create accounts and assign roles.

If you set `SEED_DEMO_USERS=0` before the first start, no accounts are created and there is no way in
through the UI — accounts registered through the web *Register* form get the Normal User role only, and
promoting one to Administrator requires an existing administrator. Leave `SEED_DEMO_USERS=1` for the first
start, sign in as `admin`, create your own administrator account from **Admin → Users & roles**, and then
delete or disable the demo accounts.

## 14. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| **Address already in use** | another process holds port 5000 — change `PORT` in `.env` |
| **Microphone will not start** | open the site as `http://127.0.0.1:5000` or over HTTPS; a LAN IP over plain HTTP is blocked by the browser |
| **GTM result says "unavailable"** | Admin → Models: GTM must be enabled, the three files present in `gtm_model/model/`, and the label check must pass |
| **GTM labels rejected** | the model's labels differ from `config/classes.json` — export a model with the same 17 labels in the same order |
| **MP3 / M4A upload fails to decode** | reinstall `imageio-ffmpeg`; on Linux also `sudo apt-get install libsndfile1` |
| **`soundfile` fails to import on Linux** | `sudo apt-get install libsndfile1` |
| **First prediction is very slow** | expected on a cold start — set `WARMUP_MODELS=1` to pre-load the model |
| **Upload rejected as too large** | raise `MAX_UPLOAD_MB` in `.env`, then raise the matching limit in Admin → Settings |
| **Want a clean database** | stop the app, delete `database/sonicsentinel.db`, start again |
| **Tests fail with a missing Node.js error** | install Node.js, or ignore — those GTM comparison tests skip themselves without it |
