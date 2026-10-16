# Team Contribution Record

**Team:** SFC-FutureMind · **Project:** SonicSentinel AI
**Members:** 4

Source of truth for the day-by-day split: `SonicsSentinal-Task Assign Sheet in 5 Days.xlsx` in this folder.
This document restates that sheet against the actual folders and files in the repository, so each member can
be questioned on code they own.

---

## Ownership summary

| Member | Modules owned (folders / files) | Key contributions | Can explain (evaluator questions) |
|---|---|---|---|
| **Haleema** | `scripts/build_dataset.py`, `scripts/generate_synthetic_dataset.py`, `scripts/dataset_report.py`, `feature_extraction/features.py`, `feature_extraction/build_features.py`, `python_models/train.py`, `python_models/evaluation.py`, dashboard analytics | Dataset collection, organisation, metadata verification and duplicate removal; the 193-feature extractor; training of the first Python model; evaluation harness producing accuracy / precision / recall / F1 / confusion matrix; dashboard analytics and charts; dataset and model evidence packaging | What the 193 features are and why; how the 70/15/15 split avoids source leakage; how the confusion matrix is produced; why macro F1 is the selection metric |
| **Huma Kiran** | `audio_preprocessing/` (`loader.py`, `preprocess.py`, `noise_reduction.py`, `quality.py`, `validation.py`, `fingerprint.py`), `python_models/ensemble.py`, `src/services/alerts_service.py`, `src/services/review_service.py` | Pre-processing pipeline — resampling, normalisation, trimming, validation; audio-quality assessment; alternative Python models 2 and 3 for the comparison; Python vs GTM comparison and top-3 confidence handling; alerts, alert history and the manual review workflow; end-to-end integration testing and bug fixing | Why noise reduction is skipped on stationary signals; how Good / Acceptable / Poor / Unusable are decided; how the ensemble weights were chosen; when a record enters the review queue |
| **Taha Mubashir Khan** | `gtm_model/` (`pipeline.py`, `inference.py`, `export_gtm_samples.py`), `scripts/train_gtm_model.py`, `static/js/gtm.js`, `static/js/live.js`, `static/js/capture-worklet.js`, `feature_extraction/gtm_spectrogram.py`, `templates/`, `documentation/` | GTM dataset split verification, independent training and testing, prediction integration; live microphone detection with 2-second windows; waveform, spectrogram and 3D spectrum UI; reports and export; detection history with search and filter; landing page, navigation and responsive layout; architecture documentation and screenshots | Why the model runs in the browser and not on the server; how `gtm.js` and `pipeline.py` are proven to compute the same input; why transfer-only training scored 59.9% against 68.0% fine-tuned; how the live window smoothing works |
| **Atiq Imran** | `src/__init__.py`, `src/security.py`, `src/blueprints/`, `database/schema.sql`, `alert_rules/` (`engine.py`, `rules.py`, `alert_rules.json`), `src/services/live_service.py`, `Dockerfile`, `render.yaml`, `wsgi.py` | Backend and database structure; authentication and RBAC; upload API, pre-processing integration and DB storage; the decision / rule engine — confidence, severity, repeated detection, critical-event rules; security, audit trail, model version tracking, admin functionality; deployment, final QA, `AI_USAGE.md` and GitHub cleanup | The permission matrix and why each role lands where it does; how a rule turns a prediction into an alert; what the top-two margin and consecutive-detection requirements are for; how escalation is configured |

---

## Day-by-day assignment (as recorded in the task sheet)

| | **Haleema** | **Huma Kiran** | **Taha Mubashir Khan** | **Atiq Imran** |
|---|---|---|---|---|
| **Day 1 — Dataset & Foundation** | 10 sound classes ka dataset collect/organize, metadata verify, duplicates remove | Audio preprocessing pipeline: resampling, normalization, trimming, validation | Existing UI/landing page finalize, navigation & responsive layout | Backend/database structure, authentication + RBAC setup |
| **Day 2 — ML Development** | Python Model 1: feature extraction (MFCC/Mel etc.) + training | Python Model 2 & 3: train alternative models for comparison | GTM model: dataset split verify, independent training/testing, prediction integration | Upload Audio API + preprocessing/backend integration + DB storage |
| **Day 3 — Core Detection System** | Python models evaluation: Accuracy, Precision, Recall, F1, confusion matrix | Python vs GTM comparison, Top-3 predictions + confidence handling | Live microphone detection, 1–3 sec audio windows, waveform/spectrogram UI | Decision/rule engine: confidence, severity, repeated detection, critical-event rules |
| **Day 4 — Full Application Features** | Dashboard analytics, charts, model-performance results | Alerts + alert history + manual review workflow | Reports/export, detection history, search/filter functionality | Security, audit trail, model version tracking, user/admin functionality |
| **Day 5 — Testing & Final Delivery** | Dataset/model evidence + unseen/noisy audio testing | Complete frontend/backend integration testing + bug fixing | Documentation: architecture, flow diagrams, screenshots, README | Deployment, final QA, demo preparation, AI_USAGE.md + GitHub cleanup |

---

## Codebase map

| Area | Files | Owner |
|---|---|---|
| Audio input and validation | `audio_preprocessing/loader.py`, `validation.py` | Huma Kiran |
| Pre-processing and quality | `audio_preprocessing/preprocess.py`, `noise_reduction.py`, `quality.py`, `fingerprint.py` | Huma Kiran |
| Features | `feature_extraction/features.py`, `build_features.py`, `live_render.py` | Haleema |
| Dataset and augmentation | `scripts/build_dataset.py`, `generate_synthetic_dataset.py`, `synthetic/`, `import_urbansound8k.py`, `relabel_dataset.py`, `select_sample_audio.py`, `augmentation/` | Haleema |
| Python model | `python_models/train.py`, `train_cnn.py`, `cnn.py`, `ensemble.py`, `inference.py`, `evaluation.py` | Haleema (train, evaluation) · Huma Kiran (ensemble) |
| GTM model | `gtm_model/`, `feature_extraction/gtm_spectrogram.py`, `scripts/train_gtm_model.py`, `static/js/gtm.js` | Taha Mubashir Khan |
| Decision engine and rules | `alert_rules/engine.py`, `rules.py`, `alert_rules.json` | Atiq Imran |
| Web app and roles | `src/__init__.py`, `src/security.py`, `src/blueprints/`, `src/services/` | Atiq Imran |
| Alerts and review | `src/services/alerts_service.py`, `review_service.py` | Huma Kiran |
| Live monitoring | `static/js/live.js`, `capture-worklet.js`, `src/services/live_service.py` | Taha Mubashir Khan (client) · Atiq Imran (server) |
| UI | `templates/`, `static/css/`, `static/js/` | Taha Mubashir Khan |
| Tests and documentation | `tests/`, `documentation/`, `reports/` | shared — each member wrote the tests for their own modules |

---

## Declaration

Each member confirms that they engineered, understood and tested the modules listed against their name, and
is prepared to explain and defend those implementations to the evaluation committee.

| Member | Signature | Date |
|---|---|---|
| Haleema | | |
| Huma Kiran | | |
| Taha Mubashir Khan | | |
| Atiq Imran | | |
