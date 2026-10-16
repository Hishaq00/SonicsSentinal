# Hidden-test readiness checklist

SRS 1.8-9 says the evaluators will test with unseen recordings that may contain the conditions below. For each
one: how SonicSentinel handles it, where that is tested, and the known risk.

| Condition | How the system handles it | Evidence | Known risk |
|---|---|---|---|
| Background noise | Training data at 0–35 dB SNR with 8 background types; augmentation adds more; adaptive noise reduction; quality analysis measures the noise floor and flags "excessive noise" | `test_preprocessing_quality.py::test_quality_detects_excessive_noise`, `test_noise_reduction_*`; noise-robustness in `reports/python_model/*/noise_robustness.json` | Below ~5 dB SNR accuracy drops; such clips are marked Poor and routed to review |
| Echo / reverberation | Rooms (small room, hall, factory) and reverb RT60 0.2–0.8 s in the data and augmentation | dataset metadata `environment` column | Very long reverb (large halls) is not in the data |
| Low volume | Peak normalisation; quiet live windows are boosted like the browser does; "low signal" quality flag; near-silent files rejected with a message | `test_audio_io.py::test_near_silent_file_rejected`, `test_quality_low_signal` | Very distant events close to the noise floor may be classified Unknown |
| Live microphone (not an uploaded file) | Both models are trained on a live rendering of every training recording — microphone band limit and self-noise, the browser's gain control, per-window normalisation with no trimming — so the live monitor is not weaker than upload analysis | `test_dataset_tools.py::test_live_rendering_matches_the_live_service_path`; live-rendering rows in `reports/python_model/*/summary.md` and `reports/gtm_model/*/summary.md` | A microphone very different from the four simulated profiles |
| Everyday sounds that are not events (camera shutter, keyboard, door click, calm speech, empty room) | Trained as Background Noise hard negatives after a camera shutter was classified Gunshot at 94 % | `reports/python_model/*/summary.md` (Background Noise recall); dataset `variant` values `everyday-clicks/*`, `empty-room*`, `neutral-speech` | Background Noise is still the weakest class for both models |
| Different devices | Device simulation (phone, laptop, headset, CCTV, body camera, studio) in data and augmentation; any sample rate 8–192 kHz is resampled | `test_audio_io.py` (sample rates, subtypes, stereo) | Unusual microphones with strong filtering |
| Partial events | 2 s segments with 1 s hop; event segments chosen by level; clips shorter than 2 s are padded | `test_short_clip_becomes_one_padded_segment`, `test_event_segment_selection_drops_quiet_segments` | A cut-off event with no attack can be confused |
| Overlapping sounds | Flagged when two non-background classes both get ≥ 0.25; routed to review with both class names | `test_alert_engine.py::test_overlapping_sounds_detected` | Overlaps are flagged, not separated into two events |
| Similar categories (gunshot vs backfire vs fireworks vs explosion vs door slam, scream vs shouting, alarm vs horn, drill vs machinery fault, help vs speech) | Fireworks, Vehicle Backfire, Explosion, Door Impact and Drilling or Grinder Sound are now their own classes, so these sounds are named instead of falling into Gunshot or Machinery Fault; two independent models must agree for Gunshot; top-two margin check; similar top scores → review | `test_alert_engine.py` (margin, agreement tests); per-class results in `reports/python_model/*/summary.md` | Explosion and Fireworks still overlap with Gunshot in the GTM model (Explosion precision 0.54); Normal Machinery is still not a class |
| Re-encoded files (MP3, OGG, M4A, FLAC, other bit depths) | Content-based format detection, FFmpeg fallback, near-duplicate fingerprint | `test_audio_io.py::test_compressed_formats_decode`, `test_security_db_live.py::test_reencoded_copy_flagged_as_near_duplicate` | FFmpeg must be installed for MP3 / M4A |
| Distant sound sources | Distance simulation 0.5–30 m in data, 2–25 m in augmentation | dataset metadata `distance_m` | Beyond ~30 m untested |
| Silent / damaged / unsupported files | Rejected with a clear reason | `test_audio_io.py` (silent, empty, truncated, fake audio, unsupported extension, too short/long, size, channels, sample rate) | — |

## Before the demo

- [ ] `python -m pytest` passes (see `reports/test_results.txt`)
- [ ] `python_models/saved/active.json` points to the intended model; Admin → Models shows it
- [ ] GTM model loads in the browser (Admin → Models → GTM status; the live monitor shows GTM results)
- [ ] FFmpeg installed (`ffmpeg -version`) so MP3 / M4A decode
- [ ] Microphone permission allowed in the browser; test on the live monitor
- [ ] Evaluator and admin accounts work (README credentials)
- [ ] One file of every class from `sample_audio/` gives the expected result (17 files, refreshed with `python -m scripts.select_sample_audio`)
- [ ] A silent file and a renamed text file are rejected
- [ ] Alert rules at their defaults (Admin → Alert rules → version history) unless a change is being demonstrated
- [ ] Surprise-modification rehearsal: change a threshold (Admin → Settings / Alert rules), add an allowed
      format (Admin → Settings), change segment duration (Settings → Segments), add a filter (history filters in
      `src/services/analytics.py::apply_filters`), add a category (`config/classes.json` + data + retrain both models)
