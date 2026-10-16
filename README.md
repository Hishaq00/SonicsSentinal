# SonicSentinel AI — Real-Time Acoustic Threat Intelligence Platform

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![Framework Flask](https://img.shields.io/badge/Framework-Flask_3.1-black.svg)](https://flask.palletsprojects.com/)
[![Machine Learning Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn_|_XGBoost-orange.svg)](https://scikit-learn.org/)
[![Edge AI TensorFlow.js](https://img.shields.io/badge/Edge_AI-TensorFlow.js_|_GTM-FF6F00.svg)](https://teachablemachine.withgoogle.com/)
[![Tests](https://img.shields.io/badge/Tests-see_reports%2Ftest__results.txt-brightgreen.svg)](reports/test_results.txt)
[![License MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **TechWiz 7 — NextWave AI and ML Competition Entry**  
> *Theme:* AcousticX Intelligence  
> *Target Domain:* Emergency Threat Recognition, Public Safety & Industrial Machinery Health Monitoring  

---

## 📖 Project Overview

**SonicSentinel AI** is an intelligent, dual-model acoustic event detection and real-time audio threat intelligence system. Traditional security systems rely on surveillance cameras that fail in low-light, smoky, or obstructed environments. SonicSentinel AI listens to the acoustic environment to identify critical events — such as **Gunshots, Glass Breaking, Panic Screams, Machinery Faults, and Distress Calls** — before optical sensors detect them.

### 🌟 Key Differentiators:
1. **Dual-Model Consensus Architecture:** Combines a server-side **193-feature Python Machine Learning Ensemble** with a client-side **Google Teachable Machine (GTM) Deep CNN** running via TensorFlow.js in the operator's browser.
2. **Deterministic Critical-Event Rule Engine:** Multi-window debounce, top-two margin thresholds, and confidence verification eliminate false alarms.
3. **Comprehensive Audio Quality Assessment:** Real-time analysis of digital clipping, silence, SNR, and noise floor.
4. **Human-in-the-Loop Review Queue:** Automatic routing of model disagreements and uncertain cases to human operators with non-destructive override auditing.
5. **Role-specific dashboards:** review desk, security console, maintenance console and administrator system overview, in a responsive dark/light UI.

---

## 🎯 Supported Sound Categories (10 Mandatory + 7 Optional Classes)

| # | Sound Category | Code | Target Severity | Default Action & Protocol |
|:---:|:---|:---:|:---:|:---|
| 1 | **Machinery Fault** | `MCH` | High | Maintenance alert; inspect bearings, belts, and motor balance. |
| 2 | **Glass Breaking** | `GLS` | High | Security perimeter breach; activate CCTV and notify patrol. |
| 3 | **Alarm or Siren** | `ALM` | High | Evacuation alert; cross-reference facility fire/security panels. |
| 4 | **Vehicle Horn** | `HRN` | Low / Med | Environmental traffic logging. |
| 5 | **Animal Sound** | `ANM` | Low | Contextual event; perimeter wildlife/pet tracking. |
| 6 | **Gunshot** | `GUN` | **Critical** | Armed threat protocol; lockdown recommendation; camera focus. |
| 7 | **Panic Scream** | `SCR` | **Critical** | Human safety distress; immediate guard dispatch. |
| 8 | **Aggression** | `AGR` | High / Crit | Violent conflict; security intervention required. |
| 9 | **Person Asking for Help** | `HLP` | **Critical** | Distress phrases (*"Help me"*, *"Emergency"*); emergency response. |
| 10 | **Background Noise** | `BGN` | Info | Ambient noise floor (rain, HVAC, traffic hum, empty rooms, everyday clicks, calm speech); hidden by default. |
| 11 | **Drone Sound** *(optional)* | `DRN` | Medium | Security alert; locate the drone and check whether it is authorised. |
| 12 | **Drilling or Grinder Sound** *(optional)* | `DRL` | Low | Check the work is scheduled; after hours treat it as a possible break-in. |
| 13 | **Fireworks** *(optional)* | `FWK` | Medium | Check permission and fire risk; keeps fireworks apart from gunfire. |
| 14 | **Vehicle Backfire** *(optional)* | `BKF` | Low | Logged only; keeps exhaust backfires apart from gunshots. |
| 15 | **Explosion** *(optional)* | `EXP` | **Critical** | Emergency protocol: alert emergency services, evacuate, check for fire. |
| 16 | **Door Impact** *(optional)* | `DOR` | Medium | Check for forced entry; escalates to High when the banging repeats. |
| 17 | **Graffiti Spray** *(optional)* | `GRF` | Medium | Possible vandalism; check cameras and send a patrol. |

---

## 🏛️ System Architecture

```
[ Uploaded Audio Clip ]   OR   [ Live Microphone (1.0s - 3.0s Windows) ]
                  │
                  ▼
   [ 1. Ingestion & Audio Validation ] ──> Reject if silent/clipped/corrupt
                  │
                  ▼
   [ 2. Python Preprocessing (22.05 kHz, Mono, Normalise, NR) ]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 [ 3A. 193 Features ]   [ 3B. GTM 43×232 Mel Spectrogram ]
        │                   │
        ▼                   ▼
 [ Python Ensemble ]    [ Google Teachable Machine (TF.js) ]
 (XGBoost + MLP + HGB)  (Client-Side Deep CNN)
        │                   │
        └─────────┬─────────┘
                  ▼
 [ 4. Consensus & Confidence Comparison ] ──> |P_Py - P_GTM|
                  │
                  ▼
 [ 5. Critical-Event Rule Engine ] ──> Multi-window confirmation
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
  [ Alerts ]  [ Review ]  [ History & Timeline ]
```

---

## ⚡ Quickstart & Installation Instructions

### Prerequisites
* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
* **Python Version:** Python 3.10 to 3.14
* **Browser:** a current Chrome, Edge or Firefox (the GTM model runs in the browser with TensorFlow.js)
* **Node.js (optional):** runs the GTM browser-parity tests in `tests/test_gtm.py`; without it they are skipped

### 1. Clone the Repository
```bash
git clone https://github.com/Hishaq00/SFC-FutureMind.git
cd "SonicSentinel AI"
```

### 2. Create and Activate Virtual Environment
**On Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**On Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Development Server
```bash
python run.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🔑 Evaluator & Demo Credentials

The database is pre-seeded with role-segregated evaluation accounts:

| Username | Password | Role | Lands on | Permissions / Purpose |
|:---|:---|:---|:---|:---|
| **`admin`** | `Admin@2026` | **Administrator** | System overview | Full platform access, rule editing, evaluation runs, user management |
| **`reviewer`** | `Review@2026` | **Audio Reviewer** | Review desk | Manual review queue access, category override, listening tools |
| **`security`** | `Secure@2026` | **Security Operator** | Security console | Live microphone monitoring, alert acknowledgement, escalation |
| **`maintenance`**| `Maintain@2026`| **Maintenance** | Maintenance console | Machinery fault alarms, sensor health inspection |
| **`evaluator`** | `Evaluate@2026`| **Competition Evaluator**| System overview | Dedicated benchmark evaluation account |
| **`user`** | `User@2026` | **Normal User** | Overview | Personal audio uploads, waveform viewing, report downloads |

Each role opens its own dashboard at `/dashboard`. The security and maintenance consoles list the alerts for the categories whose rule names that role in `responsible_roles` (Alert rules page), with acknowledge / escalate / dismiss / resolve buttons. Operators can still open the site-wide overview at `/overview`.

---

## 🧪 Comprehensive Walkthrough (Execution Instructions)

### 1. Audio Upload & Analysis
* Navigate to **Analyze Audio** (`/analyze`).
* Drag-and-drop any audio clip from the [`sample_audio/`](sample_audio/) folder (e.g. `sample_gunshot.wav`).
* Click **Analyze Recording**.
* **Observe:** Waveform viewer, Mel Spectrogram, signal quality score, Python 193-feature classification, independent GTM client-side prediction, consensus badge, and alert triggers.

### 2. Live Microphone Monitoring
* Click **Live Monitor** (`/live`).
* Click **Start Monitoring** and grant browser microphone permission.
* Speak or play test sounds. Observe rolling audio windows (1.0s to 3.0s), real-time FFT visualizers, and immediate classification badges.

### 3. Chronological Timeline & History
* Open **Timeline** (`/timeline`) to view compact, color-coded chronological sound events.
* Open **Event History** (`/history`) to filter by ID, date, confidence range, severity, and export records directly to **CSV** or **Excel**.

### 4. Manual Review Queue
* Disagreed or uncertain events are routed to **Review Queue** (`/review`).
* Reviewers can listen to audio segments, view dual-model probability deltas, and override classifications while preserving original audit logs.

### 5. Running Automated Tests
The full suite (audio I/O, pre-processing, quality, features, Python model, GTM, alert engine, comparison,
roles, review, security, database, duplicates, live microphone):
```bash
python -m pytest
```
The latest result is in `reports/test_results.txt`. Performance against the SRS targets:
```bash
python -m scripts.measure_performance      # writes reports/performance.md
```

### 6. Retraining the Python Model
Training compares SVM, Random Forest, Histogram Gradient Boosting, XGBoost and MLP with a grouped
cross-validated search on the training split, builds a soft-voting ensemble of the three best on the validation
split and tunes the critical-class decision weights on the validation split. The test split is only used for
the reported numbers.
```bash
python -m feature_extraction.build_features   # 193 features per 2 s segment -> data/features/
python -m python_models.train                 # new version in python_models/saved/, report in reports/python_model/<version>/
pip install -r requirements-gtm.txt           # PyTorch, needed for the CNN training only
python -m python_models.train_cnn             # log-mel CNN member; saved only if it beats the feature model on validation
```
Administrators can switch the active version on the **Models** page.

### 7. Retraining the GTM Model
The GTM model is the Teachable Machine audio network (Google's speech-commands base + a new softmax layer), trained on the same splits, pre-processing and evaluation as the Python model, on exactly the input `static/js/gtm.js` computes in the browser:
```bash
pip install -r requirements-gtm.txt      # PyTorch, needed for training only
python -m scripts.train_gtm_model        # writes gtm_model/model/ and reports/gtm_model/<version>/
```

---

## 📁 Repository Structure

```
├── AI_USAGE.md                 # Mandatory competition AI usage declaration form
├── README.md                   # Complete system manual & execution instructions
├── LICENSE                     # MIT Open-Source License
├── requirements.txt            # Python dependencies
├── run.py                      # Application launcher (http://127.0.0.1:5000)
├── config/                     # Classes and configuration single source of truth
├── audio_dataset/              # 4,921 clips + 2,100 augmented (train/validation/test), not in git: see its README
├── sample_audio/               # one example recording per category (17) for evaluators
├── audio_preprocessing/        # DSP resampling, normalization, noise gating, quality
├── feature_extraction/         # 193 handcrafted acoustic feature extractor
├── augmentation/               # Acoustic transforms (pitch, time, reverb, noise)
├── python_models/              # Trained weighted soft-voting ensemble & inference
├── gtm_model/                  # Google Teachable Machine exported TF.js model files
├── alert_rules/                # Configurable JSON alert engine & multi-window debounce
├── documentation/              # 2,000+ word Technical Blog, Project Report, SRS Matrix
├── notebooks/                  # Interactive EDA and feature engineering notebooks
├── reports/                    # 100-sample comparison CSV report, confusion matrices
├── screenshots/                # HD screen captures of all interfaces
├── src/                        # Flask application factory, blueprints, and models
├── static/                     # CSS, JS, Three.js 3D canvas, and Web Audio scripts
├── templates/                  # Jinja2 templates (Landing, Dashboard, Live, Timeline)
└── tests/                      # Pytest automated test suite
```

---

## 📚 Documentation

| Document | Contents |
|---|---|
| [Project report](documentation/PROJECT_REPORT.md) | requirements traceability, architecture, database design, DFD / use case / activity / sequence / decision diagrams, dataset, models, evaluation, testing, security, privacy, limitations |
| [Technical blog](documentation/TECHNICAL_BLOG.md) | 2,000+ word write-up |
| [Hidden-test checklist](documentation/HIDDEN_TEST_CHECKLIST.md) | readiness for noisy, distant, re-encoded, overlapping and similar sounds |
| [Submission checklist](documentation/SUBMISSION_CHECKLIST.md) | status of every SRS deliverable |
| [Development log](documentation/DEVELOPMENT_LOG.md) · [Team contributions](documentation/TEAM_CONTRIBUTIONS.md) | SRS 1.8 records |
| [AI usage](AI_USAGE.md) | AI tools declaration |
| [Model comparison report](reports/model_prediction_and_confidence_comparison_summary.md) | 100 unseen recordings, Python vs GTM |
| [Test results](reports/test_results.txt) | full pytest run — 166 passed, 8 skipped, 0 failed |
| [Performance measurements](reports/performance.md) | SRS 1.7-1 upload and live-window timings |
| [NFR compliance](documentation/NFR_COMPLIANCE.md) | measured result against every non-functional target |

---

## 🔗 Submission Links

| Item | Link |
|---|---|
| Public GitHub repository | https://github.com/Hishaq00/SFC-FutureMind |
| Deployed application | https://wa.sarimtools.com/ |
| Project report | [documentation/PROJECT_REPORT.md](documentation/PROJECT_REPORT.md) |
| Technical blog | [documentation/TECHNICAL_BLOG.md](documentation/TECHNICAL_BLOG.md) |
| Demonstration video | _not yet recorded — the shot list is in `17 - Demonstration Video/RECORDING_BRIEF.md`; the `.mp4` link goes here once recorded_ |

---

## 📌 Assumptions

The application was designed and evaluated under the following assumptions. They
are stated here because several of them bound how the results should be read.

1. **Controlled, ethical testing only.** This is a competition prototype, not a
   certified emergency-response or law-enforcement system. No alert it raises
   should be treated as a dispatch decision.
2. **A modern desktop browser is available.** The Teachable Machine model runs in
   the browser with TensorFlow.js, so a current Chrome, Edge or Firefox is
   required for the two-model comparison. Without it the app degrades to the
   Python model alone and labels the result *Single-Model Result*.
3. **The microphone is reachable over HTTPS or localhost.** Browsers refuse
   microphone access on a plain-HTTP LAN address, so live monitoring assumes the
   site is opened at `127.0.0.1` locally or over HTTPS in deployment.
4. **Audio arrives as a file or a live window, not as a stream from fixed
   sensors.** There is no hardware sensor integration; every input is an upload
   or a browser microphone capture.
5. **A 2-second segment is enough to identify an event.** Both models classify
   2-second segments (live windows overlap, 2 s every 1 s). Events shorter than
   roughly 200 ms, or ones whose evidence is spread over much longer than 2 s,
   are outside what the current segmentation captures well.
6. **The dataset is predominantly synthetic and procedurally generated**, with
   ethically licensed UrbanSound8K recordings mixed in. It approximates real
   deployment audio; it is not a recording of any one real site, and accuracy in
   a specific environment may differ.
7. **Speech samples are synthetic.** The *Person Asking for Help* class uses
   text-to-speech renderings of the five SRS-defined safety phrases, so accent,
   language and vocal-stress coverage is narrower than real human speech.
8. **One machine, single-node deployment.** Performance figures were measured on
   one laptop-class CPU with no GPU, and the scalability target was verified
   against a single SQLite/PostgreSQL node.
9. **Class labels are mutually exclusive at decision time.** Overlapping sounds
   are *detected and reported*, but the final decision still names one category.
10. **Evaluators use the seeded demo accounts.** Roles, permissions and dashboards
    assume the accounts created by `SEED_DEMO_USERS=1` on first start.

Known limitations are listed in
[documentation/PROJECT_REPORT.md](documentation/PROJECT_REPORT.md) and
[documentation/NFR_COMPLIANCE.md](documentation/NFR_COMPLIANCE.md).

---

## 🔒 Security, Privacy & Ethics

* **Privacy First:** Microphone monitoring requires explicit browser consent and displays a prominent glowing red indicator whenever active.
* **No Cloud Audio Leaks:** Audio inference is performed 100% on-premise/locally. Audio is never sent to third-party generative-AI cloud APIs.
* **Competition Disclaimer:** SonicSentinel AI is a research and competition prototype developed for controlled testing and is not certified for direct integration with 911 dispatch systems.

---

## 👥 Authors & Team Contributions

* **Competition:** TechWiz 7 (Aptech Limited)  
* **Team:** SFC-FutureMind — Haleema, Huma Kiran, Taha Mubashir Khan, Atiq Imran  
* **Module ownership:** [documentation/TEAM_CONTRIBUTIONS.md](documentation/TEAM_CONTRIBUTIONS.md)  
* **Repository:** [GitHub Repository](https://github.com/Hishaq00/SFC-FutureMind)  
* **Date:** September 2026  
