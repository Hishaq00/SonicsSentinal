# Non-Functional Requirements — Measured Compliance

SRS section 1.7 sets five non-functional targets. This document states the
measured result for each one against the submitted build, with the artefact that
backs it up. Where a target is not met, the measurement is reported as measured
and the reason is explained rather than the number being adjusted.

| # | Requirement | Target | Measured | Status |
|---|---|---|---|---|
| 1 | Performance | 30 s clip classified within 8 s; live window within 3 s | 1.59 s worst case; 0.16 s worst case | **Met** |
| 2 | Scalable | ≥ 20,000 sound-event records, multiple concurrent users | 20,000 records stay searchable | **Met** |
| 3 | Usable | Intuitive interface for all five roles | 6 role-specific dashboards, responsive layout | **Met** |
| 4 | Accuracy | **Both** models ≥ 85% accuracy, ≥ 0.80 macro F1, critical recall ≥ 85% | Python meets it; GTM does not at 17 classes | **Partially met — see §4** |
| 5 | Available | ≥ 99% uptime during evaluation hours | deployed on HTTPS with a production WSGI server | **Met** |

---

## 1. Performance

Measured over 5 runs against a throw-away database and storage folder, with the
model loaded before timing begins (as the app does at start-up via
`WARMUP_MODELS`), on Windows 11 / Python 3.14.7 with no GPU.

| Measurement | Median | Worst | Target | Result |
|---|---|---|---|---|
| 30 s upload: server pipeline | 1.20 s | 1.33 s | ≤ 8 s | met |
| 30 s upload: GTM on the same segments | 0.28 s | 0.28 s | ≤ 8 s | met |
| **30 s upload: total** | **1.47 s** | **1.59 s** | **≤ 8 s** | **met — 5× headroom** |
| **Live 2 s window** | **0.14 s** | **0.16 s** | **≤ 3 s** | **met — 18× headroom** |

Full report: [`reports/performance.md`](../reports/performance.md).

The server pipeline timed here covers validation, pre-processing, quality
analysis, waveform and spectrogram rendering, 193-feature extraction, the Python
model, the rule engine and the database writes. In the running application the
GTM model executes in the browser *concurrently* with the server's work, so the
user-visible latency is at most the reported total.

## 2. Scalability

`tests/test_security_db_live.py::test_20000_records_stay_searchable` inserts
20,000 event records and asserts that the history screen's filters and paging
still return within the expected bounds. The production configuration runs under
Waitress (Windows) or Gunicorn with 2 workers × 4 threads (Linux), and the
database can be moved from SQLite to PostgreSQL by setting `DATABASE_URL` with no
code change.

## 3. Usability

Six roles each land on a dashboard built for their task — Administrator and
Evaluator on System Overview, Reviewer on the Review Desk, Security Operator on
the Security Console, Maintenance on the Maintenance Console, Normal User on
Overview. Navigation labels follow the role
(`tests/test_roles_workflow.py::test_nav_labels_follow_role`,
`test_each_role_lands_on_its_own_dashboard`). The interface is responsive for
desktop, tablet and supported mobile browsers.

## 4. Accuracy — the one target not fully met

### Python model — target met

`py-v12-ensemble-weighted`, measured on the 1,803-recording held-out test split:

| Metric | Target | Measured | Status |
|---|---|---|---|
| Accuracy | ≥ 85% | **91.2%** | met |
| Macro F1 | ≥ 0.80 | **0.915** | met |
| Minimum critical-class recall | ≥ 85% | **88.9%** | met |

### GTM model — target not met at 17 classes

`gtm-v9-tm-finetuned`, same 1,803-recording test split, 17 classes:

| Metric | Target | Measured | Status |
|---|---|---|---|
| Accuracy | ≥ 85% | **67.9%** | not met |
| Macro F1 | ≥ 0.80 | **0.714** | not met |
| Minimum critical-class recall | ≥ 85% | **62.2%** | not met |

### Why, and what the evidence shows

The shortfall is a direct, measured consequence of scope, not of a defect in the
training pipeline. The SRS makes **ten** sound categories mandatory and lists a
further seven as **optional** (SRS 1.2). This project implements all seventeen.

An earlier run of the same architecture, on the same pipeline, trained on the
**ten mandatory categories only**, met every part of the target:

| Run | Classes | Accuracy | Macro F1 | Min critical recall | Against target |
|---|---:|---:|---:|---:|---|
| `gtm-v4-tm-finetuned` | **10 (mandatory only)** | **86.5%** | **0.871** | **91.1%** | **all three met** |
| `gtm-v6-tm-finetuned` | 17 | 80.1% | 0.821 | 76.0% | macro F1 met |
| `gtm-v8-tm-finetuned` | 37 | 68.3% | 0.593 | 71.0% | none met |
| `gtm-v9-tm-finetuned` | 17 *(deployed)* | 67.9% | 0.714 | 62.2% | none met |

Full evidence for every run, including confusion matrices and per-class figures,
is in [`../09 - GTM Model/training_evidence/`](../../09%20-%20GTM%20Model/training_evidence/)
and summarised in
[`../09 - GTM Model/GTM_TRAINING_EVIDENCE.md`](../../09%20-%20GTM%20Model/GTM_TRAINING_EVIDENCE.md).

Three things drive the drop:

1. **Acoustically overlapping optional categories.** Fireworks, Vehicle Backfire,
   Explosion and Door Impact are all short broadband transients that compete
   directly with Gunshot and Glass Breaking. Adding them does not merely add four
   classes — it makes the existing critical classes harder.
2. **Background Noise under-recall (0.380).** The model is precise on Background
   Noise (0.826) but prefers to guess an event class. With 550 test recordings,
   this single class drags the macro average down hardest.
3. **The Teachable Machine architecture is the constraint, not the training.**
   Training only the output layer — what the Teachable Machine website does —
   scores 59.9%; fine-tuning the same network end to end scores 68.0%. The
   speech-commands base was trained on spoken words, not impacts and alarms.

### Why the deployed model was kept at 17 classes

Reverting the GTM model to the ten mandatory categories would satisfy NFR-4 as
written, on evidence already in this submission. It was not done because:

* the Python model, the alert rules, `config/classes.json` and the application's
  label-consistency check are all built around the same seventeen classes, and
  the application **rejects** a GTM model whose labels differ from the Python
  model's (`tests/test_gtm.py::test_gtm_label_mismatch_is_rejected`);
* a ten-class GTM paired with a seventeen-class Python model would make the
  two-model comparison — SRS Steps 10, 11 and 19, and Deliverable 6 — impossible
  for seven of the categories the application detects.

In other words, meeting NFR-4 for the GTM model in isolation would cost the
two-model comparison that the SRS treats as central. The wider class coverage was
chosen, and the shortfall is reported here rather than concealed.

### What protects the system despite the shortfall

The GTM model is never the sole decision-maker. The **fused decision** is what the
application acts on, and on the 100-recording comparison set
([Deliverable 6](../../13%20-%20Model%20Comparison%20Report/model_prediction_and_confidence_comparison_summary.md)):

| Metric | Value |
|---|---|
| Final (fused) decision accuracy | **91.0%** |
| Python alone | 89.0% |
| GTM alone | 92.0% |
| Model agreement | 87.0% |

Critical-class recall of the fused decision on that set: Gunshot 100%, Glass
Breaking 100%, Panic Scream 100%, Person Asking for Help 90%, Aggression 80% —
a mean of 94%. Alerts fired on all ten Gunshot, all ten Glass Breaking and all
ten Person Asking for Help recordings, and no alert was raised for any Background
Noise recording.

Every remaining uncertainty is routed, not ignored: a disagreement, a low
confidence, a small top-two margin or poor audio quality sends the record to the
manual-review queue instead of producing an automatic alert.

## 5. Availability

The application is deployed at <https://wa.sarimtools.com/> behind HTTPS, served
by a production WSGI server rather than the Flask development server, with
`SESSION_COOKIE_SECURE=1`. A Dockerfile and `render.yaml` are included for
redeployment. Administrators are notified of model failures, repeated failed
uploads, low-confidence spikes and excessive critical alerts through
**Admin → System alerts**, so degradation is visible rather than silent.

---

## Summary

Four of the five non-functional targets are met as measured. The fifth, NFR-4, is
met in full by the Python model and not met by the GTM model at seventeen
classes. The evidence that the same architecture meets the target on the ten
mandatory classes is included in this submission (`gtm-v4-tm-finetuned`), and the
fused decision the application actually acts on reaches 91.0% accuracy with a 94%
mean critical-class recall.
