# Development log

SRS 1.8-3: work completed, problems encountered, dataset changes, model failures, code changes, tests
performed. One entry per day; each member adds their own lines. Entries below were written from the git
history, sprint logs and the reports — team members should add what they did personally.

---

## Day 1–2 (up to 25 Sep 2026)

**Work completed**
- Flask application with authentication, roles, upload, live monitoring, alerts, review, history, analytics,
  admin (commit `56a385a` "Initial commit").
- Dataset built: UrbanSound8K import + procedural and TTS synthesis → 4,921 originals, 2,100 augmented,
  70/15/15 split (`data/dataset_statistics.json`, generated 25 Sep 09:02 UTC).
- Python models trained and compared: `py-v1` (HGB) → `py-v2` (ensemble) → `py-v3`, `py-v4` (ensemble +
  critical-class weighting, active since 25 Sep 09:21 UTC).
- Landing page, GTM model integration (commit `ae2ab7c`).

**Problems / model failures**
- Python test accuracy stayed at 83.4 % (target 85 %); Machinery Fault recall 44 %.
- GTM export v3 scored 38 % inside the app (found on day 3).

**Tests performed** — `pytest` suites for audio I/O, pre-processing, quality, features, model, GTM.

_[members: add your own entries]_

---

## Day 3 (26 Sep 2026)

**Work completed**
- GTM root cause found: training used the first second of each file while the browser classifies the loudest
  second; training used raw audio while the app sends pre-processed audio; the old comparison report's GTM
  column was not produced by the GTM model.
- Browser-identical GTM input in Python; NumPy runner for TF.js models; GTM fine-tuned end to end
  (`gtm-v4-tm-finetuned`, 86.5 % test accuracy, 64 min CPU); comparison report regenerated with the real model.
- UI redesign (identity, landing page, animation layer, 3D spectrum views).

**Dataset changes** — none (same splits reused for GTM).

**Model failures** — Teachable Machine "transfer" training (frozen base) reached only 77.9 %; the previous
export 37.9 %. Remaining weak spot: sirens read as vehicle horns.

**Tests performed** — 4 new GTM tests (browser parity, exported model); headless Chrome screenshots.

_[members: add your own entries]_

---

## Day 4 (27 Sep 2026)

**Work completed**
- Separate dashboards per role: review desk, security console, maintenance console; admin lands on the system
  overview.
- Alert action buttons shared between the alerts page and the consoles.
- New tests: role workflow (25), alert engine (34), security / database / duplicates / live microphone.
- `scripts/measure_performance.py` for the SRS performance targets.
- Project report, technical blog and AI_USAGE.md rewritten with the real numbers; hidden-test and submission
  checklists.

**Problems encountered / code changes**
- Truncated WAV files passed validation (libsndfile reports the shortened length) → header check added in
  `audio_preprocessing/loader.py`.
- A test for the rules editor overwrote the real `alert_rules/alert_rules.json` because the file path was a
  default argument; the file was restored from git and the path is now resolved at call time.
- `test_event_segment_selection_drops_quiet_segments` used a 15 dB window for a 12 dB burst → test corrected.
- Timing tests are only meaningful with the model already loaded; they now warm up first.

**Model work** — Python model retrained with a larger hyper-parameter search. The retrain did not move the
numbers much: the model was not the problem, the labels were (see Day 5).

**Tests performed** — full suite, see `reports/test_results.txt`.

_[members: add your own entries]_

---

## Day 5 (27 Sep 2026, evening) — the labels were wrong, and the optional categories

**Why this day happened.** Testing with sounds from outside the dataset (files the user downloaded
themselves) showed failures the test-split numbers did not: an empty room was called Machinery Fault, a
camera shutter was called Gunshot at 94 %, a horse neigh came out as Unknown Sound, and a vehicle-horn clip
split the two models (Python said Person Asking for Help, GTM said Gunshot). The 83.4 % test accuracy was
real but it was measured on a dataset that contained the same label mistakes as the model.

**Dataset changes**
- Label corrections (`scripts/relabel_dataset.py`, old Audio IDs kept in `notes`): UrbanSound8K
  `air_conditioner` + `engine_idling` (150 clips) Machinery Fault -> Background Noise; `drilling` +
  `jackhammer` (150) -> the new Drilling or Grinder Sound; synthetic Background clips containing a door
  close or knock (33) -> Door Impact.
- All **7 SRS optional categories** added as real classes: Drone Sound, Drilling or Grinder Sound,
  Fireworks, Vehicle Backfire, Explosion, Door Impact, Graffiti Spray. New generators in
  `scripts/synthetic/optional_sources.py`; 420 more real drilling/jackhammer clips imported from many
  source recordings (`import_urbansound8k --only ... --per-source 4`).
- Hard negatives that were missing entirely: empty rooms (mains hum, rumble, hiss), everyday clicks (camera
  shutters, light switches, keyboards, pens, staplers), calm neutral speech -> Background Noise; horse,
  big-cat, wolf and elephant calls -> Animal Sound.
- Dataset: 4,921 -> **7,866 originals**, 17 classes, 5,190 augmented copies (now one per training
  recording, not only for the generated ones).
- Alert rules for the seven new categories (`alert_rules/alert_rules.json`, Explosion is Critical with
  model agreement required; Door Impact escalates to High when the banging repeats).

**Code changes**
- `feature_extraction/live_render.py` (new): renders a dataset recording the way the **live microphone**
  path delivers it — microphone band limit and self-noise, presence peak, distance, the browser's gain
  control, 22.05 kHz, then per-window pre-processing with no trimming. `build_features.py` now writes both
  renderings and marks each row; training uses both, selection and headline numbers use the upload
  rendering, and the live rendering is reported separately. Until now the Python model had never seen
  live-style audio, which is why the live monitor was weaker than the upload page.
- Confidence calibration (temperature scaling) added to `python_models/evaluation.py` and used by both
  training scripts; for GTM the temperature is folded into the output layer so the browser shows calibrated
  confidences. GTM test-window ECE 0.053 -> 0.017.
- `import_urbansound8k.py`: `--only/--count/--per-source` round robin over source recordings, and it merges
  into `incoming/metadata.csv` instead of overwriting it.
- Bug found and fixed: `time_shift` augmentation could shift a clip shorter than 0.5 s entirely out of the
  frame, producing silent training copies.
- Bug found and fixed: `build_features` did not include the class label in its cache fingerprint, so a
  relabel did not trigger a rebuild.

**Model failures / findings**
- The Machinery Fault recall of 0.44 was a labelling artefact, not a model weakness. After the correction
  it is **0.889** with no change to the algorithm.
- Teachable Machine's own training procedure (frozen base) reaches only 69.6 % on 17 classes against 80.4 %
  for the fine-tuned network, so the website-trained model is not usable as the app's model.
- Going from 10 to 17 classes costs GTM accuracy (0.865 -> 0.803); it is now the weaker of the two models
  and below the 85 % target. The Python model is above it.

**Process problem** — multiple training runs were executing concurrently on the same working copy
(primary terminal, a background worker process, and an interactive test shell). A stray `train_gtm_model --no-finetune`
run exported a 69.4 % transfer model over the fine-tuned export at 17:56; it was replaced by
`gtm-v6-tm-finetuned` minutes later. A Python model was also activated that was the weakest of three on
validation. Lesson: one person runs training at a time, and `active.json` is set from the validation
result, never from the test result.

**Tests performed** — `tests/test_dataset_tools.py` (new, 16 tests): optional-category generators, variant
generators, the time-shift fix, the relabel plan, temperature scaling, and a parity test asserting that a
window from `live_render` is identical to what `live_service` computes.

_[members: add your own entries]_
