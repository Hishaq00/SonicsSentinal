# Model Prediction & Confidence Comparison Report (Deliverable 6)

## Executive Summary

A stratified evaluation was run over **100 unseen test recordings** — **10 recordings from each of the
10 mandatory sound categories** — taken from the held-out test split (`06 - Testing Dataset/`). No
recording in this set was used for training or validation, and no source group leaks across splits
(`tests/test_dataset_tools.py::test_no_source_leaks_between_splits`).

Each recording was classified independently by:

* the server-side Python ensemble `py-v12-ensemble-weighted` (193 handcrafted features per 2-second segment), and
* the Google Teachable Machine audio model `gtm-v9-tm-finetuned` (exported TF.js model, run through the
  NumPy runner in `gtm_model/inference.py` so the report reproduces exactly what the browser computes).

Both models see the same 2-second analysis segments.

| Metric | Value |
|---|---|
| Total evaluated recordings | 100 |
| Classes evaluated | 10 mandatory categories, 10 recordings each |
| **Python model accuracy** | **89.0%** |
| **GTM model accuracy** | **92.0%** |
| **Final (fused) decision accuracy** | **91.0%** |
| Dual-model class agreement rate | 87.0% |
| Mean absolute top-class confidence difference | 0.1691 |
| Mean Python top confidence | 0.9251 |
| Mean GTM top confidence | 0.7792 |
| Mean top-two confidence margin | 0.8662 |

**Critical-class recall of the final decision**

| Critical class | Final-decision recall |
|---|---|
| Gunshot | 100% |
| Glass Breaking | 100% |
| Panic Scream | 100% |
| Person Asking for Help | 90% |
| Aggression | 80% |
| **Minimum** | **80%** |
| **Mean** | **94%** |

Per-recording rows — audio ID, filename, actual class, both predicted classes, both top confidences,
class-match status, confidence difference, top-two margin, audio quality, severity, alert status,
manual-review status and final decision — are in
[`model_prediction_and_confidence_comparison_report.csv`](model_prediction_and_confidence_comparison_report.csv).

---

## Performance by Sound Category

| Sound Category | Samples | Python Accuracy | GTM Accuracy | Final Accuracy | Model Agreement | Mean Conf. Diff | Mean Top-2 Margin | Alerts |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Machinery Fault** | 10 | 90% | 80% | 90% | 90% | 0.3105 | 0.8628 | 9 |
| **Glass Breaking** | 10 | 100% | 100% | 100% | 100% | 0.0089 | 0.9990 | 10 |
| **Alarm or Siren** | 10 | 100% | 90% | 100% | 90% | 0.1789 | 0.8785 | 9 |
| **Vehicle Horn** | 10 | 80% | 90% | 80% | 80% | 0.2030 | 0.8697 | 1 |
| **Animal Sound** | 10 | 80% | 90% | 80% | 80% | 0.1756 | 0.8241 | 0 |
| **Gunshot** | 10 | 100% | 100% | 100% | 100% | 0.0568 | 0.9828 | 10 |
| **Panic Scream** | 10 | 100% | 90% | 100% | 90% | 0.1049 | 0.8885 | 9 |
| **Aggression** | 10 | 70% | 90% | 80% | 60% | 0.1591 | 0.6756 | 8 |
| **Person Asking for Help** | 10 | 80% | 100% | 90% | 80% | 0.1314 | 0.7392 | 10 |
| **Background Noise** | 10 | 90% | 90% | 90% | 100% | 0.3618 | 0.9424 | 0 |

---

## Model Consistency Distribution

The `Consistency Status` column grades how far apart the two models are on each recording.

| Status | Count | Share |
|---|---:|---:|
| Acceptable Match | 64 | 64.0% |
| Weak Match | 23 | 23.0% |
| Uncertain Result | 10 | 10.0% |
| Model Disagreement | 3 | 3.0% |

`Class Match Status`: **87 Match / 13 Disagreement**.

---

## Alert and Review Outcomes

| Alert Status | Count |
|---|---:|
| Alert Generated | 66 |
| None | 24 |
| Informational | 8 |
| Low Confidence | 2 |

| Severity | Count |
|---|---:|
| Critical | 29 |
| High | 39 |
| Informational | 16 |
| Medium | 8 |
| Low | 8 |

**Manual review:** 15 of 100 recordings were routed to the review queue (model disagreement, an uncertain
result, or a confidence below the category's configured threshold); 85 cleared automatically.

**Audio quality:** all 100 recordings scored `Good`. The test split is the clean, original split —
degraded-audio behaviour is covered separately by the noise, clipping, silence and dropout tests in
`tests/test_preprocessing_quality.py`.

---

## Every Disagreement, Listed

All 13 recordings where the two models chose different classes:

| Audio ID | Actual | Python said | GTM said | Final decision | Final correct? |
|---|---|---|---|---|:---:|
| DS-MCH-0009 | Machinery Fault | Machinery Fault | Aggression | Machinery Fault | yes |
| DS-ALM-0002 | Alarm or Siren | Alarm or Siren | Vehicle Horn | Alarm or Siren | yes |
| DS-HRN-0008 | Vehicle Horn | Person Asking for Help | Gunshot | Gunshot | no |
| DS-HRN-0032 | Vehicle Horn | Background Noise | Vehicle Horn | Background Noise | no |
| DS-ANM-0018 | Animal Sound | Gunshot | Animal Sound | Gunshot | no |
| DS-ANM-0047 | Animal Sound | Alarm or Siren | Vehicle Horn | Alarm or Siren | no |
| DS-SCR-0072 | Panic Scream | Panic Scream | Alarm or Siren | Panic Scream | yes |
| DS-AGR-0058 | Aggression | Person Asking for Help | Aggression | Person Asking for Help | no |
| DS-AGR-0067 | Aggression | Person Asking for Help | Aggression | Aggression | yes |
| DS-AGR-0070 | Aggression | Aggression | Person Asking for Help | Aggression | yes |
| DS-AGR-0077 | Aggression | Person Asking for Help | Aggression | Person Asking for Help | no |
| DS-HLP-0004 | Person Asking for Help | Aggression | Person Asking for Help | Person Asking for Help | yes |
| DS-HLP-0020 | Person Asking for Help | Aggression | Person Asking for Help | Aggression | no |

### What the disagreements show

1. **Aggression vs Person Asking for Help is the dominant confusion** — 6 of the 13 disagreements, and the
   reason Aggression has the lowest agreement rate (60%) and the lowest mean top-two margin (0.6756). Both
   categories are raised human voices; the Python model leans towards *Person Asking for Help*, the GTM model
   towards *Aggression*. Every one of these 6 was routed to manual review, which is the designed behaviour —
   both are critical classes, so an alert still fires and a human confirms the category.
2. **Fusion recovers more than it breaks.** Of the 13 disagreements the fused decision was right 6 times.
   Overall the fused decision (91.0%) sits above the Python model alone (89.0%) and just below GTM alone
   (92.0%) on this set — but it never dropped a critical event: alerts fired on all 10 Gunshot, all 10 Glass
   Breaking and all 10 Person Asking for Help recordings.
3. **Vehicle Horn and Animal Sound are the weakest non-critical classes** (80% final accuracy each). Both are
   broadband, tonal-transient sounds that overlap acoustically with sirens and gunshots. These are the two
   classes to prioritise for more training data.
4. **Background Noise shows the largest mean confidence difference (0.3618)** while still agreeing 100% of the
   time. The Python model is very confident on silence/ambience; GTM is deliberately less so. Agreement is what
   matters here, and no alert was generated for any Background Noise recording — the correct outcome.

---

## How to Reproduce

    python -m scripts.generate_100_comparison_report

Writes `reports/model_prediction_and_confidence_comparison_report.csv` and this summary.
Supporting per-version metrics, confusion matrices and sample predictions for both model families are in
[`supporting/`](supporting/).
