"""Generate the Model Prediction and Confidence Comparison Report (SRS Deliverable 6).

At least 100 unseen test recordings and at least 10 per category: 10 test recordings of every class
(17 classes since 27 Sep 2026 -> 170 recordings).

Both models see each recording the way the web application analyses an upload:
pre-processing, 2 s segments every 1 s, and the recording-level probabilities
are the mean over the event segments (alert_rules.engine.decide_recording).
The Python ensemble classifies each segment's features; the GTM model is the
exported TensorFlow.js model in gtm_model/model/, run on exactly the input
static/js/gtm.js computes in the browser (gtm_model/pipeline.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from audio_preprocessing.loader import decode_audio  # noqa: E402
from audio_preprocessing.preprocess import preprocess_audio  # noqa: E402
from audio_preprocessing.quality import analyze_quality  # noqa: E402
from config import DATASET_DIR, METADATA_CSV, PROJECT_ROOT, class_names  # noqa: E402
from gtm_model import pipeline as gtm_pipeline  # noqa: E402
from gtm_model.inference import GTMClassifier  # noqa: E402
from python_models.evaluation import CRITICAL_CLASSES  # noqa: E402
from python_models.inference import get_classifier  # noqa: E402


def mean_scores(score_list: list[dict[str, float]], classes: list[str]) -> dict[str, float]:
    return {c: float(np.mean([s[c] for s in score_list])) for c in classes}


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_CSV)
    test_df = df[(df["split"] == "test") & (df["status"] == "original")]
    clf = get_classifier()
    gtm = GTMClassifier()
    classes = class_names()

    rows = []
    print(f"Generating the comparison: 10 test recordings x {len(classes)} classes "
          f"(Python {clf.version}, GTM {gtm.version})...")

    for cls in classes:
        sub = test_df[test_df["class_label"] == cls].head(10)
        for _, item in sub.iterrows():
            audio_id = item["audio_id"]
            rel_path = item["relative_path"]
            full_path = DATASET_DIR / rel_path

            # 1. Load audio and assess quality
            decoded = decode_audio(full_path)
            y = decoded.mono()
            sr = decoded.sample_rate
            prep = preprocess_audio(y, sr)
            q = analyze_quality(prep.y, prep.sr)

            # 2. Segments exactly as the app cuts an upload; recording level = mean over the event segments
            audio = gtm_pipeline.upload_audio(decoded.samples, sr)
            segments = gtm_pipeline.app_segments(audio)
            events = [s for s, is_event in segments if is_event] or [s for s, _ in segments]

            # 3. Python inference (193 features per 2 s segment)
            py_probs = mean_scores(clf.predict_segments([s.audio for s in events]), classes)
            sorted_py = sorted(py_probs.items(), key=lambda x: x[1], reverse=True)
            py_pred = sorted_py[0][0]
            py_conf = float(sorted_py[0][1])

            # 4. GTM inference: the exported TF.js model on the browser's 43 x 232 input for each segment
            gtm_probs = mean_scores(gtm.predict_inputs([gtm_pipeline.segment_input(audio, s)[0] for s in events]), classes)
            sorted_gtm = sorted(gtm_probs.items(), key=lambda x: x[1], reverse=True)
            gtm_pred = sorted_gtm[0][0]
            gtm_conf = float(sorted_gtm[0][1])

            # 5. Comparative Metrics
            class_match = "Match" if py_pred == gtm_pred else "Disagreement"
            conf_diff = round(abs(py_conf - gtm_conf), 4)

            py_margin = round(sorted_py[0][1] - (sorted_py[1][1] if len(sorted_py) > 1 else 0), 4)

            if class_match == "Match" and conf_diff < 0.20:
                consistency = "Acceptable Match"
            elif class_match == "Match":
                consistency = "Weak Match"
            elif conf_diff < 0.15:
                consistency = "Model Disagreement"
            else:
                consistency = "Uncertain Result"

            final_cat = py_pred if py_conf >= gtm_conf else gtm_pred
            final_conf = round(max(py_conf, gtm_conf), 4)
            is_correct = (final_cat == cls)

            # Alert and Severity assignment
            if final_cat in ["Gunshot", "Panic Scream", "Person Asking for Help"] and final_conf >= 0.70:
                sev = "Critical"
                alert_st = "Alert Generated"
            elif final_cat in ["Glass Breaking", "Alarm or Siren", "Machinery Fault", "Aggression"]:
                sev = "High"
                alert_st = "Alert Generated" if final_conf >= 0.65 else "Low Confidence"
            elif final_cat == "Vehicle Horn":
                sev = "Medium"
                alert_st = "Informational"
            else:
                sev = "Low" if final_cat == "Animal Sound" else "Informational"
                alert_st = "None"

            manual_rev = "Yes" if (consistency in ["Model Disagreement", "Uncertain Result"] or not is_correct or q["label"] in ["Poor", "Unusable"]) else "No"

            disagree_note = ""
            if class_match == "Disagreement":
                disagree_note = f"Python predicted {py_pred} ({py_conf*100:.1f}%) while GTM predicted {gtm_pred} ({gtm_conf*100:.1f}%)."

            rows.append({
                "Audio ID": audio_id,
                "Filename": Path(rel_path).name,
                "Actual Class": cls,
                "Python Predicted Class": py_pred,
                "Python Top Confidence": round(py_conf, 4),
                "GTM Predicted Class": gtm_pred,
                "GTM Top Confidence": round(gtm_conf, 4),
                "Class Match Status": class_match,
                "Consistency Status": consistency,
                "Confidence Difference": conf_diff,
                "Top-Two Margin": py_margin,
                "Audio Quality": q["label"],
                "Severity": sev,
                "Alert Status": alert_st,
                "Manual Review Status": manual_rev,
                "Final Decision": final_cat,
                "Result Status": "Correct" if is_correct else "Incorrect",
                "Major Disagreement Explanation": disagree_note
            })

    res_df = pd.DataFrame(rows)
    csv_path = reports_dir / "model_prediction_and_confidence_comparison_report.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"Successfully generated CSV at {csv_path} with {len(res_df)} rows.")

    # Generate Markdown Summary
    acc = (res_df["Result Status"] == "Correct").mean()
    py_acc = (res_df["Python Predicted Class"] == res_df["Actual Class"]).mean()
    gtm_acc = (res_df["GTM Predicted Class"] == res_df["Actual Class"]).mean()
    match_rate = (res_df["Class Match Status"] == "Match").mean()
    avg_diff = res_df["Confidence Difference"].mean()
    crit = res_df[res_df["Actual Class"].isin(CRITICAL_CLASSES)]
    crit_recall = (crit["Final Decision"] == crit["Actual Class"]).mean() if len(crit) else float("nan")

    md_content = f"""# Model Prediction & Confidence Comparison Report (Deliverable 6)

## Executive Summary
This report fulfills **TechWiz 7 SRS Deliverable 6 (Page 39)**. A stratified evaluation was conducted across **{len(res_df)} unseen test recordings** (10 clips per sound category). Each audio clip was independently evaluated by the server-side Python Classification Ensemble (193 features, `{clf.version}`) and the Google Teachable Machine (GTM) audio model (`{gtm.version}`), each on the application's 2-second analysis segments (recording result = mean over the event segments).

* **Total Evaluated Recordings:** {len(res_df)}
* **Classes Evaluated:** {len(classes)} sound categories (10 mandatory + 7 SRS optional; 10 recordings each)
* **Python Model Accuracy:** {py_acc * 100:.2f}%
* **GTM Model Accuracy:** {gtm_acc * 100:.2f}%
* **Final Decision Accuracy:** {acc * 100:.2f}%
* **Dual-Model Class Agreement Rate:** {match_rate * 100:.2f}%
* **Mean Absolute Confidence Difference ($|P_{{Py}} - P_{{GTM}}|$):** {avg_diff:.4f}
* **Critical-Class Recall of the Final Decision ({", ".join(CRITICAL_CLASSES)}):** {crit_recall * 100:.1f}%

---

## Performance by Sound Category

| Sound Category | Samples | Python Accuracy | GTM Accuracy | Final Accuracy | Model Agreement | Mean Conf. Diff | Alert Triggers |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for cls in classes:
        c_sub = res_df[res_df["Actual Class"] == cls]
        c_py = (c_sub["Python Predicted Class"] == cls).mean() * 100
        c_gtm = (c_sub["GTM Predicted Class"] == cls).mean() * 100
        c_acc = (c_sub["Result Status"] == "Correct").mean() * 100
        c_match = (c_sub["Class Match Status"] == "Match").mean() * 100
        c_diff = c_sub["Confidence Difference"].mean()
        c_alerts = (c_sub["Alert Status"] == "Alert Generated").sum()
        md_content += f"| **{cls}** | {len(c_sub)} | {c_py:.1f}% | {c_gtm:.1f}% | {c_acc:.1f}% | {c_match:.1f}% | {c_diff:.4f} | {c_alerts} |\n"

    md_content += f"""
---

## Model Consistency Distribution
* **Acceptable Match:** {(res_df["Consistency Status"] == "Acceptable Match").sum()} / {len(res_df)} ({((res_df["Consistency Status"] == "Acceptable Match").mean() * 100):.1f}%)
* **Weak Match:** {(res_df["Consistency Status"] == "Weak Match").sum()} / {len(res_df)} ({((res_df["Consistency Status"] == "Weak Match").mean() * 100):.1f}%)
* **Model Disagreement:** {(res_df["Consistency Status"] == "Model Disagreement").sum()} / {len(res_df)} ({((res_df["Consistency Status"] == "Model Disagreement").mean() * 100):.1f}%)
* **Uncertain Result:** {(res_df["Consistency Status"] == "Uncertain Result").sum()} / {len(res_df)} ({((res_df["Consistency Status"] == "Uncertain Result").mean() * 100):.1f}%)

The complete recording-by-recording breakdown is stored in `reports/model_prediction_and_confidence_comparison_report.csv`.
"""
    summary_path = reports_dir / "model_prediction_and_confidence_comparison_summary.md"
    summary_path.write_text(md_content, encoding="utf-8")
    print(f"Generated Markdown Summary at {summary_path}")

if __name__ == "__main__":
    main()
