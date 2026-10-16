# Execution

## 1. Start it

    Windows:        .venv\Scripts\python.exe run.py
    Linux / macOS:  .venv/bin/python run.py

Wait for:

    SonicSentinel AI running on http://127.0.0.1:5000  (Ctrl+C to stop)

Open **http://127.0.0.1:5000**. Use `127.0.0.1`, not the machine's LAN address — browsers only allow the
microphone on localhost or over HTTPS. Change the port with `PORT=5001` in `.env`.

Installation steps: `14 - Installation Instructions/INSTALLATION.md`.
Live deployment and evaluator credentials: `16 - Deployment URL/DEPLOYMENT.md`.

## 2. Registration and login

**Register** — press *Register* in the header (`/register`), then give a username, email and password.
The password must be **at least 8 characters and contain both a letter and a digit**, and is stored hashed,
never in plain text (`tests/test_security_db_live.py::test_passwords_are_hashed`).

Every new account is created as a **Normal User**, whatever role was asked for: it can upload and analyse
its own audio, but cannot see other users' records, review, or reach the consoles. The registration form
does let you *request* a higher role — the request is logged to the audit trail and sent to an
administrator, who grants it from **Admin → Users & roles**. Nobody can self-assign a role
(`test_normal_user_cannot_change_roles`).

**Login** — `/login`. Five failed attempts in a row lock the account temporarily
(`test_account_locks_after_repeated_failures`). After signing in you land on the dashboard that belongs to
your role — each role has its own (`test_each_role_lands_on_its_own_dashboard`).

**Seeded accounts**

| Username | Password | Role |
|---|---|---|
| admin | Admin@2026 | Administrator — everything |
| evaluator | Evaluate@2026 | Competition Evaluator — evaluation runs |
| reviewer | Review@2026 | Audio Reviewer — manual review queue |
| security | Secure@2026 | Security Operator — live monitoring and alerts |
| maintenance | Maintain@2026 | Maintenance Personnel — machinery-fault alerts |
| user | User@2026 | Normal User — own uploads only |

Change your own password at **Profile → Change password** (`/profile`).

## 3. Audio upload

**Analyze audio** (`/analyze`) — drag a file in or browse for one, then press **Analyze**.

* Accepted formats: **WAV, MP3, FLAC, OGG, M4A/AAC**
* Size limit: `MAX_UPLOAD_MB` in `.env` (default 30 MB)
* Rejected on upload: empty files, near-silent files, sample rates below the minimum, and files whose
  contents are not really audio (a renamed `.txt` is refused, not crashed on)
* A file that has already been uploaded is flagged as a **duplicate** by acoustic fingerprint — a
  re-encoded or volume-adjusted copy is still caught

Sample clips to try are in `sample_audio/` (and in `16 - Deployment URL/sample_audio/`).

Operators can upload several files at once through **batch upload**; the batch page (`/batches/<code>`)
shows progress and results.

## 4. Audio preview

On the record page (`/records/<audio_id>`) the **player at the top** streams the audio back. It offers:

* **Original** — the file exactly as uploaded
* **Processed** — what the model actually heard, after resampling, trimming, normalisation and
  noise reduction

Play both to hear what pre-processing changed. On the live monitor, each captured window can be played back
from the per-window table in the same way.

## 5. Live microphone monitoring

**Live monitor** (`/live`) → **Start monitoring** → allow the microphone when the browser asks. Consent is
recorded before any capture starts (`test_microphone_needs_consent_first`).

Audio is captured in **2-second windows every 1 second**. Each window is classified by both models,
compared, and checked against the alert rules. Because the windows overlap, the displayed result is the
**mean of the recent windows**, so one odd window cannot flip the category on its own.

On screen: the live waveform, the scrolling spectrogram, the 3D spectrum, and a per-window table with both
models' verdicts. Press **Stop** to end the session; the session and its windows are saved.

A quicker look: press **Listen live** in the dashboard header to watch the waveform react to the room
without starting a full monitoring session.

## 6. Metadata viewing

The **Metadata** panel on the record page shows what was read from the file and measured from it:

| Field | Example |
|---|---|
| Audio ID | `DS-GUN-0042` |
| Filename and format | `gunshot_sample.wav`, WAV |
| Duration | 2.59 s |
| Sample rate / channels / bit depth | 22050 Hz, mono, 16-bit |
| File size | 114 KB |
| SHA-256 | the content hash used for duplicate detection |
| Uploaded by / uploaded at | user and timestamp |
| Segments analysed | how many 2-second windows the models saw |

Dataset-wide metadata for all **17,056** dataset entries (13,443 train including augmented copies, 1,810
validation, 1,803 test) is at **`/dataset`**, and downloadable as CSV from **`/dataset/metadata.csv`**. The
same file ships in `07 - Dataset Metadata/`.

## 7. Waveform viewing

The **waveform** is drawn on the record page under the player. It shows the amplitude envelope over time
with the **analysed segment highlighted** — the window the models actually scored, which is centred on the
loudest part of the event rather than on the start of the file. Silent lead-in and trailing silence are
visibly outside the highlight.

## 8. Spectrogram viewing

The **spectrogram** sits next to the waveform: frequency against time, brightness for energy. It is
generated server-side and also served on its own at `/records/<audio_id>/spectrogram.png`.

This is the same mel representation the GTM model is fed, so what you see is close to what the model sees —
a gunshot appears as a bright vertical stripe, an alarm as horizontal bands, background noise as an even
wash.

## 9. Python prediction

The **Python** panel is filled first, by the server. It is the `py-v12-ensemble-weighted` model: 193
handcrafted features per 2-second segment (MFCC, mel, spectral, chroma, f0 and more), fed to a weighted
ensemble of HistGradientBoosting + Random Forest + MLP.

It shows the predicted class, its confidence, and a confidence bar for **every one of the 17 classes**, so
you can see the runner-up as well as the winner.

## 10. GTM prediction

The **GTM** panel fills a moment later. The Google Teachable Machine model is **not** run on the server —
it is downloaded into your browser and run there with TensorFlow.js, on the same 2-second segments. This is
why the tab must stay open until the GTM result arrives.

It shows the same thing: predicted class, confidence, and a bar for every class.

If GTM cannot load, the page says so and falls back to the Python result alone, marked
**Single-Model Result** — it does not fail the analysis.

## 11. Confidence interpretation

Confidence is the model's probability for the class it chose, from 0 to 1. Two numbers matter alongside it:

* **Top-two margin** — the gap between the best and second-best class. A high confidence with a *small*
  margin means the model is torn between two similar classes, which is weaker evidence than the headline
  number suggests.
* **Confidence difference** — how far apart the two models' top confidences are.

The **Consistency status** on each record combines all of this:

| Status | What it means |
|---|---|
| **Acceptable Match** | both models chose the same class, both above the minimum confidence, and their confidences are within `max_confidence_difference` (0.25 by default) — the strongest result |
| **Weak Match** | same class, but one model was below the confidence threshold or the two confidences are far apart |
| **Model Disagreement** | different classes, both confident — goes to manual review |
| **Uncertain Result** | at least one model was below the confidence threshold — goes to manual review |
| **Single-Model Result** | GTM was unavailable; the Python model answered alone |

If the fused confidence falls below the configured `unknown_threshold`, the record is reported as
**"Unknown Sound"** rather than guessing.

## 12. Model comparison

The **Comparison** panel puts both models side by side: each one's top class, each one's confidence, the
class-match status, the confidence difference and the top-two margin — then the **final decision**, which is
a weighted mean of the two probability vectors, not simply one model overriding the other.

The two models are genuinely independent: the Python prediction is never an input to the GTM scores
(`test_python_prediction_is_not_an_input_to_gtm_scores`).

**Admin → Models** shows both models' versions and runs the label check that confirms the GTM labels still
match the Python model's classes. A mismatch is what stops the two results from being comparable, and the
app refuses such a model.

The full 100-recording comparison is in `13 - Model Comparison Report/`.

## 13. Audio-quality interpretation

Every upload is scored for quality before anything is classified. The verdict appears on the record page
with the reasons behind it.

| Label | Meaning | Effect |
|---|---|---|
| **Good** | clean signal, adequate level, no clipping or dropouts | no restriction |
| **Acceptable** | usable, but noisy or quiet | most rules still allow an alert |
| **Poor** | heavy noise, clipping, or a very low signal | rules with `min_audio_quality: Acceptable` will not raise an alert — e.g. Gunshot |
| **Unusable** | silence, or a truncated/corrupt stream | not classified at all; the upload is rejected |

Detected conditions include **silence, low signal, excessive noise, severe clipping, dropouts** and a
**truncated stream**. The point is that a critical alert is never raised on evidence the system knows is
bad — a poor-quality clip that looks like a gunshot goes to manual review instead.

## 14. Alert acknowledgement

Open **Alerts** (`/alerts`), or the **Security Console** / **Maintenance Console** for role-filtered views —
a security operator sees only security categories, maintenance only machinery faults, and neither can act
on the other's alerts.

Each alert offers:

| Action | Effect |
|---|---|
| **Acknowledge** | you have seen it and are handling it; it stops being unattended |
| **Escalate** | raise the severity and notify per the rule's `escalation` block |
| **Dismiss** | a false alarm — the record is automatically sent to the review queue so the miss is examined |
| **Resolve / Close** | the situation is dealt with; the alert is closed |

Every action is written to the audit trail with who did it and when, and the record, alert, review and audit
entries stay linked (`test_record_alert_review_audit_are_linked`).

## 15. Manual review

**Review Desk** (`/review`) — for the `reviewer` and `admin` roles. A record lands here automatically when:

* the two models disagreed,
* the result was uncertain or below the confidence threshold,
* the audio quality was too poor for the category's rule, or
* an operator dismissed the alert as a false alarm.

The reviewer plays the audio, reads both models' full class probabilities, and either confirms the category
or overrides it. **An override never deletes the model outputs** — the original predictions are kept
alongside the human decision, which is what makes the review data usable for retraining later
(`test_reviewer_override_keeps_model_outputs`).

## 16. Dashboard access

Each role has its own dashboard and its own navigation; you land on it after signing in.

| Role | Dashboard |
|---|---|
| Administrator | System Overview — `/dashboard` |
| Competition Evaluator | System Overview |
| Audio Reviewer | Review Desk |
| Security Operator | Security Console |
| Maintenance Personnel | Maintenance Console |
| Normal User | Overview — `/overview` |

**Analytics** (`/analytics`) has the charts: detections over time, class distribution, severity breakdown,
model agreement rate and model performance. Operators can always open the site overview as well.

## 17. Event-history search

**Event history** (`/history`) lists every detection, newest first. Filter and search by:

* sound category
* severity (Critical / High / Medium / Low / Informational)
* alert status and review status
* date range
* free-text search on audio ID and filename
* model-agreement status

Filters combine, and results stay paged and responsive at scale — there is a test that keeps **20,000
records** searchable (`test_20000_records_stay_searchable`).

## 18. Report export

| Export | Where | Formats |
|---|---|---|
| **Event history** | the Export button on `/history` — exports the *filtered* list, not everything | CSV, Excel |
| **Single record report** | the Download report button on a record page | PDF |
| **Evaluation run report** | `/evaluation/<code>` | CSV, Excel |
| **Dataset metadata** | `/dataset` | CSV |
| **Alert rules** | Admin → Alert rules → Download | JSON |

Admins can export the audit trail from `/audit`.

## 19. Automated-test execution

    .venv\Scripts\python.exe -m pytest tests -q

A single suite, with names:

    .venv\Scripts\python.exe -m pytest tests/test_gtm.py -v

| Suite | Covers |
|---|---|
| `test_audio_io.py` | format decoding, boundary and negative cases, fake/empty/silent files |
| `test_preprocessing_quality.py` | noise reduction, clipping, silence, dropouts, truncation, quality labels |
| `test_features_model.py` | feature shape, determinism, the Python model's outputs |
| `test_cnn.py` | the CNN variant |
| `test_gtm.py` | GTM export, browser/Python parity, label mismatch, fallback |
| `test_alert_engine.py` | thresholds, consecutive detections, model agreement, severity, escalation, overlapping sounds |
| `test_roles_workflow.py` | RBAC, dashboards, review workflow, acknowledgement |
| `test_security_db_live.py` | password hashing, CSRF, open redirect, session isolation, database scale, live sessions |
| `test_dataset_tools.py` | audio-ID uniqueness, split balance, no source leakage between splits |

Some GTM comparison tests need Node.js; without it they skip themselves rather than fail.

The pre-demo checklist is `03 - Complete Source Code/documentation/HIDDEN_TEST_CHECKLIST.md`.

## 20. Retraining

    python -m feature_extraction.build_features
    python -m python_models.train

    python -m pip install -r requirements-gtm.txt
    python -m scripts.train_gtm_model            # 30-60 min on a laptop CPU
    python -m scripts.train_gtm_model --resume   # continue after an interruption

    python -m scripts.generate_100_comparison_report

The application picks up new model files by itself — reload the page, no restart.

## 21. Production

    Windows:  .venv\Scripts\waitress-serve.exe --host 0.0.0.0 --port 8000 wsgi:app
    Linux:    .venv/bin/gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:8000 wsgi:app
    Docker:   docker build -t sonicsentinel . && docker run -p 8000:8000 -e SECRET_KEY=... sonicsentinel

Serve over HTTPS and set `SESSION_COOKIE_SECURE=1`. See `16 - Deployment URL/`.

## 22. If something goes wrong

* **Address already in use** — change `PORT` in `.env`.
* **GTM result says unavailable** — Admin → Models: GTM must be enabled, the files present in
  `gtm_model/model/`, and the label check must pass.
* **Microphone will not start** — open the site as `http://127.0.0.1:5000` or over HTTPS.

The full table is in `14 - Installation Instructions/INSTALLATION.md` § 14.
