# Final submission checklist (SRS 1.10-16)

Status as of 29 Sep 2026. "Team" items cannot be produced by code and must be done by the team.

| # | Deliverable | Status | Where / what is left |
|---|---|---|---|
| 1 | Project report | **done** | `documentation/PROJECT_REPORT.md` — 7,185 words; all required diagrams and sections present; team named |
| 2 | Public GitHub URL | **done** | https://github.com/Hishaq00/SFC-FutureMind — confirm it is public |
| 3 | Complete source code | **done** | repository |
| 4–6 | Training / validation / testing datasets | **done** | `04 - Training Dataset/` (8,405), `05 - Validation Dataset/` (1,810), `06 - Testing Dataset/` (1,803); ~1.9 GB, kept out of git — see `audio_dataset/README.md` |
| 7 | Dataset metadata | **done** | `07 - Dataset Metadata/` — metadata CSV (17,056 rows), statistics, data dictionary, quality report |
| 8 | Python model | **done** | `08 - Python Model/` — saved model, card, training scripts; evidence for the active version in `13 .../supporting/python_model/py-v12-ensemble-weighted/` |
| 9 | GTM model | **done (files + training evidence)** / **team (website screenshots)** | `09 - GTM Model/` — exported model, `GTM_TRAINING_EVIDENCE.md`, `training_evidence/` for six runs. **Still missing:** a Teachable Machine *project link*, screenshots of all classes and test screenshots. To produce them: `python -m gtm_model.export_gtm_samples`, build the project on teachablemachine.withgoogle.com, save screenshots to `screenshots/gtm/` |
| 10 | Pre-processing scripts | **done** | `10 - Preprocessing Scripts/` |
| 11 | Feature-extraction scripts | **done** | `11 - Feature Extraction Scripts/` — all 10 SRS features including onset strength and tempo |
| 12 | Alert rules | **done** | `12 - Alert Rules/alert_rules/alert_rules.json` — all 11 required fields, with version history |
| 13 | Model-comparison report (≥100 recordings, ≥10 per class) | **done** | `13 - Model Comparison Report/` — 100 recordings, 10 per class, plus a row-by-row list of all 13 disagreements |
| 14 | Installation instructions | **done** | `14 - Installation Instructions/INSTALLATION.md` — all 17 SRS items |
| 15 | Execution instructions | **done** | `15 - Execution Instructions/EXECUTION.md` + `README.md` — all 20 SRS items |
| 16 | Deployment URL | **done** | https://wa.sarimtools.com/ — credentials, sample audio and testing instructions in `16 - Deployment URL/` |
| 17 | Demonstration video (.mp4) | **team — NOT DONE** | must show every item in SRS 1.10-13 (login, upload, live, validation, metadata, preprocessing, waveform, spectrogram, both predictions and confidences, comparison, quality, repeated detection, critical alert, acknowledgement, review, override, dashboard, history, report, every class, low-confidence, disagreement, noisy, overlapping, real-time critical detection). Shot list: `17 - Demonstration Video/RECORDING_BRIEF.md` |
| 18 | Technical blog (≥2,000 words) | **done** | `documentation/TECHNICAL_BLOG.md` — 4,292 words; publish it and add the public link to README |
| 19 | AI_USAGE.md | **done** | tool table, scope, files, student validation, testing and verifying members all filled in |
| 20 | Team contribution record | **done** | `documentation/TEAM_CONTRIBUTIONS.md` + the task-assignment sheet in `20 - Team Contribution Record/` |
| — | Development log | **partly** | `documentation/DEVELOPMENT_LOG.md` — add each member's daily entries |
| — | Screenshots | **done (app)** / **team (GTM website)** | `screenshots/` — 17 pages across all roles; the team adds `screenshots/gtm/` |
| — | Test results | **done** | `reports/test_results.txt` — 174 tests, 166 passed, 0 failed |
| — | Performance evidence (SRS 1.7-1) | **done** | `reports/performance.md` — 1.59 s worst upload (target 8 s), 0.16 s worst live window (target 3 s) |
| — | NFR compliance (SRS 1.7) | **done** | `documentation/NFR_COMPLIANCE.md` — 4 of 5 met; NFR-4 met by the Python model, not met by GTM at 17 classes (reasoning and evidence included) |
| — | Assumptions (SRS 1.10-11) | **done** | `README.md` → *Assumptions* |
| — | Commits across all five days, from all members | **team** | every member commits their own work; do not back-date commits |

---

## What is genuinely outstanding

Only three things, and two of them cannot be produced from the codebase:

1. **Demonstration video** — must be recorded. This is a mandatory deliverable.
2. **Teachable Machine website evidence** — project link, class screenshots, test
   screenshots. The model itself, its training configuration, sample counts,
   retraining history and integration evidence are all already submitted; what is
   missing is the web-interface artefacts, because the model was trained with
   `scripts/train_gtm_model.py` rather than on the website. See
   `09 - GTM Model/GTM_TRAINING_EVIDENCE.md` §7.
3. **Development log daily entries** and **per-member commits** — team actions.

## Known SRS shortfall to be ready to explain

**NFR-4 (accuracy).** The SRS asks that *both* models reach 85% accuracy, 0.80
macro F1 and 85% critical recall. The Python model does (91.2% / 0.915 / 88.9%).
The GTM model does not at 17 classes (67.9% / 0.714 / 62.2%). The same
architecture *did* meet all three on the 10 mandatory classes
(`gtm-v4-tm-finetuned`: 86.5% / 0.871 / 91.1%), and that evidence is in the
submission. The fused decision the application acts on reaches 91.0% accuracy
with a 94% mean critical-class recall. Full reasoning:
`documentation/NFR_COMPLIANCE.md` §4.
