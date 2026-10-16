# Sample audio

One example recording per sound category for trying the app: log in, open **Analyze audio** (`/analyze`) and upload a file.

Chosen by `python -m scripts.select_sample_audio` from the **test split** (recordings no model was trained on): for each class, the recording that both models (Python `py-v7-hgb-weighted`, GTM `SonicSentinel_GTM_v6@2026-09-27T12:57:52Z`) classify correctly with the highest confidence, preferring real UrbanSound8K recordings. They are clear examples for a demonstration; the models are not right on every recording - see `reports/` for the measured accuracy.

| Category | File | Audio ID | Source | Duration (s) | Python | GTM |
|---|---|---|---|---|---|---|
| Machinery Fault | `sample_machinery_fault.wav` | DS-MCH-0121 | synthetic-procedural (loose-part-rattle) | 3.7 | 100% | 100% |
| Glass Breaking | `sample_glass_breaking.wav` | DS-GLS-0090 | synthetic-procedural (window-pane) | 2.2 | 100% | 100% |
| Alarm or Siren | `sample_alarm_or_siren.wav` | DS-ALM-0304 | urbansound8k (UrbanSound8K fold7 siren) | 4.0 | 100% | 97% |
| Vehicle Horn | `sample_vehicle_horn.wav` | DS-HRN-0444 | urbansound8k (UrbanSound8K fold3 car_horn) | 4.0 | 98% | 95% |
| Animal Sound | `sample_animal_sound.wav` | DS-ANM-0384 | urbansound8k (UrbanSound8K fold3 dog_bark) | 4.0 | 99% | 99% |
| Gunshot | `sample_gunshot.wav` | DS-GUN-0322 | urbansound8k (UrbanSound8K fold3 gun_shot) | 1.9 | 100% | 99% |
| Panic Scream | `sample_panic_scream.wav` | DS-SCR-0095 | synthetic-procedural (scream/female) | 3.1 | 100% | 100% |
| Aggression | `sample_aggression.wav` | DS-AGR-0095 | synthetic-procedural (shouting/3-voices) | 4.4 | 100% | 100% |
| Person Asking for Help | `sample_person_asking_for_help.wav` | DS-HLP-0037 | synthetic-tts (tts-help-phrase) | 2.0 | 85% | 89% |
| Drone Sound | `sample_drone_sound.wav` | DS-DRN-0072 | synthetic-procedural (quad/approach) | 5.0 | 100% | 100% |
| Drilling or Grinder Sound | `sample_drilling_or_grinder_sound.wav` | DS-DRL-0165 | urbansound8k (UrbanSound8K fold9 jackhammer; device and distance unknown (placeholders)) | 4.0 | 94% | 90% |
| Fireworks | `sample_fireworks.wav` | DS-FWK-0152 | synthetic-procedural (firecracker-string) | 4.6 | 100% | 100% |
| Vehicle Backfire | `sample_vehicle_backfire.wav` | DS-BKF-0009 | synthetic-procedural (single-pop) | 2.5 | 100% | 100% |
| Explosion | `sample_explosion.wav` | DS-EXP-0096 | synthetic-procedural (blast-distant) | 4.7 | 99% | 100% |
| Door Impact | `sample_door_impact.wav` | DS-DOR-0168 | synthetic-procedural (kicking) | 2.4 | 99% | 100% |
| Graffiti Spray | `sample_graffiti_spray.wav` | DS-GRF-0079 | synthetic-procedural (spray-bursts) | 2.7 | 100% | 100% |
| Background Noise | `sample_background_noise.wav` | DS-BGN-0415 | urbansound8k (UrbanSound8K fold5 street_music) | 4.0 | 68% | 72% |

Percentages are the confidence each model gives the correct category when the file is analysed like an upload (mean over the event segments).

## Clips for the overlapping and uncertain paths

The ten category samples above are clean single events, so they never exercise the
overlap detection or the uncertainty reasons. These six do. They are mixtures and noisy
copies of the samples above, and each one was checked through the real analysis pipeline
(both models, the alert rules and all) before being added here.

### Overlapping sounds

Two event categories at similar level, so the fused scores put more than one class above
`detection.overlap_threshold` (0.25). The result panel shows **Overlapping sounds** and lists them.

| File | Detected | Overlapping classes |
|---|---|---|
| `sample_overlapping_panic_scream_and_aggression.wav` | Panic Scream | Panic Scream, Aggression |
| `sample_overlapping_alarm_or_siren_and_vehicle_horn.wav` | Vehicle Horn | Vehicle Horn, Alarm or Siren |
| `sample_overlapping_machinery_fault_and_drilling_or_grinder_sound.wav` | Machinery Fault | Machinery Fault, Drilling or Grinder Sound |

### Uncertain results

The same events buried in noise, so confidence falls below `detection.min_confidence` and the
two models stop agreeing. The result is routed to the review queue with its reasons.

| File | Detected | Uncertainty reasons |
|---|---|---|
| `sample_uncertain_gunshot.wav` | **Unknown Sound** | low confidence, similar top classes, models disagree, unsupported pattern |
| `sample_uncertain_door_impact.wav` | Door Impact | low confidence |
| `sample_uncertain_animal_sound.wav` | Animal Sound | low confidence, models disagree |

`sample_uncertain_gunshot.wav` is the interesting one: confidence drops under
`detection.unknown_threshold`, so the application answers **Unknown Sound** instead of guessing.

Rebuild them with `scratchpad/make_test_clips.py`, which generates candidates, runs each through
the pipeline and keeps only those that actually trigger the intended behaviour.
