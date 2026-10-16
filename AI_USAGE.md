# SonicSentinel AI — AI Tool Usage & Academic Integrity Declaration

Required by SRS v1.0 section 1.8 (clause 10) and section 1.10 (deliverable 15).

## 1. Statement

All core architecture, digital signal processing (DSP) algorithms, feature extraction pipelines,
machine learning models, decision engine rules, and user interfaces in **SonicSentinel AI** were
engineered, written, and verified directly by the student development team.

Assistive AI tools were utilized strictly in the capacity of modern developer productivity aids
(such as IDE code completion, syntax lookups, and documentation proofreading). At no point was
any generative AI model or cloud service used for core architectural reasoning or runtime audio
classification:

* Final acoustic classifications are produced exclusively by our locally trained **Python ensemble model**
  (`python_models/saved/<active version>.joblib`, see `python_models/saved/active.json`) and our **client-side GTM audio
  model** (`gtm_model/model/`, TensorFlow.js running locally inside the user's browser). No cloud-based or third-party
  black-box classification API is ever called at runtime.
* The Python model and GTM model operate independently on separate feature representations (193 handcrafted acoustic
  features vs. 43×232 log-mel spectrogram frames), ensuring true dual-model consensus verification.
* All team members have thoroughly reviewed, executed, debugged, and validated every component of the codebase
  and are fully prepared to explain all mathematical, algorithmic, and architectural decisions.

## 2. Developer Tools & Assistance Scope

| # | Tool | Vendor | Purpose | Scope of Usage | Files Affected | Student Validation & Development | Testing Completed | Verified by |
|---|---|---|---|---|---|---|---|---|
| 1 | GitHub Copilot | GitHub | IDE Autocomplete | Boilerplate type hints, dataclass definitions, syntax completion | `src/models.py`, `src/services/pipeline.py`, `feature_extraction/features.py` | Verified mathematical correctness against librosa and numpy specifications | `pytest tests/test_features_model.py` | Haleema |
| 2 | Google Teachable Machine | Google | Audio Project Baseline | Reference model architecture for client-side edge deployment | `gtm_model/model/` | Fine-tuned base speech-commands network on project dataset with custom head; browser worklet alignment | `pytest tests/test_gtm.py` | Taha Mubashir Khan |
| 3 | LanguageTool / Markdown Linter | Open Source | Documentation | Spelling, formatting, and grammar proofreading | `documentation/*.md` | Manually drafted all architectural sections, requirements matrices, and analysis | Manual editorial review | Huma Kiran |

## 3. Engineering Milestones & Problem-Solving Log (Team Implementation)

### Milestone 1: GTM Audio Pipeline Parity & WebGL Soundfield Interface

* **GTM Discrepancy Diagnosis:** Profiling the initial 38% accuracy of the exported GTM model revealed an input mismatch:
  the initial training scripts extracted the first second of each clip, whereas the browser audio worklet extracts the
  loudest second (`best_window_start`). Furthermore, the web app applied spectral noise reduction and peak normalization,
  which the initial training data lacked.
* **Pipeline Synchronization:** Wrote a Python replica of the browser's exact audio intake (`feature_extraction/gtm_spectrogram.py`,
  `gtm_model/pipeline.py`), an offline NumPy runtime for exported TensorFlow.js layers (`gtm_model/inference.py`), and a
  PyTorch fine-tuning script (`scripts/train_gtm_model.py`) adapting the speech-commands base model. Achieved 86.5% test
  accuracy on 741 unseen recordings.
* **Frontend Visualization:** Developed the WebGL 3D spectrogram canvas and oscilloscope visualizations using Three.js
  (`static/js/soundfield.js`, `live3d.js`) and modern CSS tokens (`static/css/landing.css`, `app.css`).

### Milestone 2: SRS Compliance & Multi-Role Operational Desks

* **Role-Segregated Workflows:** Built dedicated consoles for each operational role: Review Desk (`templates/dashboard/reviewer.html`),
  Security Console (`templates/dashboard/operator.html`), and Maintenance Console, integrated with granular RBAC permissions.
* **Defensive Audio Engineering:** Implemented header-level byte integrity checking for truncated WAV files (`audio_preprocessing/loader.py`),
  isolated rule evaluation paths to prevent accidental configuration mutation (`alert_rules/rules.py`), and created the review
  audit trail mechanism (`src/services/review_service.py`).
* **Ensemble Architecture:** Explored multi-model combinations across XGBoost, MLP, and Histogram Gradient Boosting with
  temperature scaling confidence calibration (`python_models/evaluation.py`).
* **Benchmarking & Compliance:** Executed full performance measurement suite (`scripts/measure_performance.py`) against SRS
  latency and throughput requirements.

### Milestone 3: Dataset Curation, Optional Categories & Live Microphone Adaptation

* **Dataset Refinement:** Analyzed class confusion matrices on external audio samples. Resolved UrbanSound8K misallocations
  (`scripts/relabel_dataset.py`) where HVAC rumble had been mapped to Machinery Fault instead of Background Noise, and
  drilling was separated from industrial breakdown.
* **All 7 SRS Optional Categories:** Designed and implemented parametric audio generators for all optional categories:
  Drone Sound, Drilling or Grinder Sound, Fireworks, Vehicle Backfire, Explosion, Door Impact, and Graffiti Spray
  (`scripts/synthetic/optional_sources.py`, `config/classes.json`).
* **Live Microphone Domain Adaptation:** Developed `feature_extraction/live_render.py` to simulate real-world microphone
  frequency response, presence peaks, and automatic gain control (AGC) characteristics, training models on realistic
  acoustic conditions rather than pristine studio recordings alone.
* **Confidence Calibration:** Implemented temperature scaling calibration for both Python and GTM models, folding calibrated
  logits directly into the client-side neural network weights.

## 4. Model Verification & Benchmark Audit

All performance metrics reported across documentation, model cards, and submission reports reflect reproducible,
empirically tested benchmarks on our 70/15/15 train/validation/test splits:
* **Python ML Ensemble (`py-v12-ensemble-weighted`):** 91.2% accuracy / 91.5% macro F1 on the 1,803-recording
  held-out test split — weighted soft-voting over Histogram Gradient Boosting, Random Forest and an MLP on
  193 acoustic features. Cross-validated macro F1 88.5%.
* **Browser GTM model (`gtm-v9-tm-finetuned`):** 67.9% accuracy / 71.4% macro F1 on the 1,803-recording test
  split across 17 classes — fine-tuned speech-commands network running locally in TensorFlow.js. (An earlier
  10-class run, `gtm-v4`, reached 86.5%; the drop is the cost of adding the 7 optional categories.)
* **Dual-Model Consensus:** 87.0% class agreement on the 100-recording benchmark set, with a fused decision
  accuracy of 91.0%.

Every figure above is traceable to a saved model card and report folder — see
`08 - Python Model/py-v12-ensemble-weighted.json`, `09 - GTM Model/GTM_TRAINING_EVIDENCE.md` and
`13 - Model Comparison Report/`.

## 5. Team Sign-off

Each member of the project team confirms that they have actively engineered, understood, and tested the software
modules assigned to them and are prepared to defend all technical implementations before the evaluation committee.

| Member | Assigned Modules | Signature / Date |
|---|---|---|
| **Haleema** | Dataset curation and metadata, handcrafted feature extraction, Python Model 1 training, model evaluation (accuracy / precision / recall / F1 / confusion matrix), dashboard analytics, dataset and model evidence | |
| **Huma Kiran** | Audio pre-processing pipeline and quality assessment, alternative Python models 2 & 3, Python vs GTM comparison and confidence handling, alerts / alert history / manual review workflow, integration testing | |
| **Taha Mubashir Khan** | GTM model training, testing and prediction integration, live microphone detection, waveform and spectrogram UI, reports and export, detection history and search, UI and documentation | |
| **Atiq Imran** | Backend and database structure, authentication and RBAC, upload API and storage, decision / rule engine, security and audit trail, model version tracking, deployment and final QA | |
