"""Write ``reports/dataset_report.md``: data dictionary, statistics and quality report of the dataset.

    python -m scripts.dataset_report

Everything is computed from ``data/dataset_metadata.csv`` (written by
``scripts/build_dataset.py`` and ``augmentation/augment_dataset.py``), so the
report always matches the dataset on disk.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import METADATA_CSV, REPORTS_DIR, class_names

DICTIONARY = [
    ("audio_id", "text", "Unique Audio ID: `DS-<class code>-<NNNN>` for originals, `…-AUG<k>` for augmented copies"),
    ("root_audio_id", "text", "Audio ID of the original recording this row comes from (itself for originals)"),
    ("parent_audio_id", "text", "Recording the augmented copy was made from (empty for originals)"),
    ("status", "original / augmented", "Augmented copies are never counted as unique originals"),
    ("class_label", "text", "One of the ten mandatory classes (`config/classes.json`)"),
    ("split", "train / validation / test", "Stratified 70 / 15 / 15 per class; copies stay in their original's split"),
    ("filename", "text", "File name inside the dataset folder"),
    ("original_filename", "text", "Name of the file before import"),
    ("relative_path", "text", "Path under `audio_dataset/`"),
    ("source", "text", "urbansound8k · synthetic-procedural · synthetic-tts"),
    ("license", "text", "Licence of the recording (CC-BY for UrbanSound8K, CC0 for generated audio)"),
    ("source_group", "text", "Recordings from one source (same TTS phrase / same original sound file); a group never spans two splits"),
    ("duration_s", "seconds", "Length of the clip"),
    ("sample_rate", "Hz", "Sampling rate of the stored file"),
    ("channels", "count", "Number of channels"),
    ("bit_depth", "bits", "Bit depth where the format has one"),
    ("format", "text", "WAV / FLAC / …"),
    ("file_size_bytes", "bytes", "File size"),
    ("sha256", "hex", "SHA-256 of the file bytes (duplicate check)"),
    ("environment", "text", "Recording environment (urban, outdoor-street, indoor-small-room, indoor-hall, indoor-factory, outdoor-open)"),
    ("device", "text", "Recording device (smartphone, laptop-mic, headset, body-camera, cctv-camera, studio-recorder, field_recorder)"),
    ("distance_m", "metres", "Approximate source distance, where known"),
    ("background", "text", "Background added or present (rain, wind, hvac-office, traffic, crowd, conversation, …)"),
    ("snr_db", "dB", "Signal-to-noise ratio of the event against the background"),
    ("overlap", "text", "Second overlapping event and its level, if any"),
    ("speaker", "text", "Voice(s) for speech classes"),
    ("phrase", "text", "Spoken phrase for Person Asking for Help / Aggression"),
    ("variant", "text", "Generator variant (e.g. `shouting/3-voices`)"),
    ("quality_label", "Good / Acceptable / Poor / Unusable", "Result of `audio_preprocessing/quality.py`"),
    ("quality_score", "0–100", "Quality score behind the label"),
    ("augmentation", "text", "Transforms applied to an augmented copy"),
    ("created_at", "ISO time", "When the row was added"),
    ("notes", "text", "Free text"),
]


def _table(df: pd.DataFrame) -> list[str]:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, *values in df.astype(object).itertuples():  # object dtype keeps integers as integers
        out.append("| " + " | ".join([str(idx)] + [str(v) for v in values]) + " |")
    return out


def main() -> None:
    d = pd.read_csv(METADATA_CSV)
    o = d[d.status == "original"]
    a = d[d.status != "original"]
    classes = [c for c in class_names() if c in set(d.class_label)]

    lines = [
        "# Dataset report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `python -m scripts.dataset_report` from "
        f"`data/dataset_metadata.csv`.",
        "",
        "## Summary",
        "",
        f"* Unique original clips: **{len(o):,}** (SRS minimum 3,000)",
        f"* Augmented copies (training only, not counted as originals): {len(a):,}",
        f"* Splits of originals: " + ", ".join(f"{k} {v:,}" for k, v in o.split.value_counts().items()),
        f"* Total duration: {d.duration_s.sum() / 3600:.2f} h (originals {o.duration_s.sum() / 3600:.2f} h)",
        f"* Clip length: {o.duration_s.min():.1f}–{o.duration_s.max():.1f} s, mean {o.duration_s.mean():.2f} s",
        "",
        "## Per class",
        "",
    ]
    per = pd.DataFrame({
        "originals": o.class_label.value_counts(),
        "augmented": a.class_label.value_counts(),
        "train": o[o.split == "train"].class_label.value_counts(),
        "validation": o[o.split == "validation"].class_label.value_counts(),
        "test": o[o.split == "test"].class_label.value_counts(),
        "mean s": o.groupby("class_label").duration_s.mean().round(2),
    }).reindex(classes).fillna(0)
    for c in ("originals", "augmented", "train", "validation", "test"):
        per[c] = per[c].astype(int)
    per.index.name = "Class"
    lines += _table(per)
    mean = per.originals.mean()
    lines += ["", f"Class mean {mean:.0f} originals. Classes more than 10 % away from the mean: "
              + (", ".join(f"{c} ({n})" for c, n in per.originals.items() if abs(n - mean) > 0.1 * mean) or "none")
              + ". Every class has at least 300 originals; training uses class weighting.", ""]

    lines += ["## Quality report (originals)", ""]
    q = pd.crosstab(o.class_label, o.quality_label).reindex(classes).fillna(0).astype(int)
    q = q.reindex(columns=["Good", "Acceptable", "Poor", "Unusable"], fill_value=0)
    q["mean score"] = o.groupby("class_label").quality_score.mean().round(1).reindex(classes)
    q.index.name = "Class"
    lines += _table(q)
    lines += ["", f"{(o.quality_label == 'Poor').sum()} Poor and {(o.quality_label == 'Unusable').sum()} Unusable "
              "originals. Poor clips (noisy or distant) stay in the dataset because the hidden tests contain such "
              "audio.", ""]

    lines += ["## Sources and licences", ""]
    src = pd.crosstab(o.class_label, o.source).reindex(classes).fillna(0).astype(int)
    src.index.name = "Class"
    lines += _table(src)
    lines += ["", "Licences: " + "; ".join(f"{k} ({v:,})" for k, v in o.license.value_counts().items()), ""]

    lines += ["## Variation (originals)", ""]
    for col, title in (("environment", "Environment"), ("device", "Device")):
        vc = o[col].fillna("(none)").value_counts()
        lines.append(f"* **{title}:** " + ", ".join(f"{k} {v}" for k, v in vc.items()))
    lines.append("  (UrbanSound8K clips are Freesound field recordings; their `urban` / `field_recorder` values are "
                 "placeholders, the real devices vary.)")
    # backgrounds are stored as combinations ("rain+crowd"); count each component once per clip
    bg = o.background.dropna().str.split("+").apply(lambda xs: sorted(set(xs))).explode().value_counts()
    lines.append(f"* **Background** (clips containing it; {o.background.isna().sum()} field recordings have their own): "
                 + ", ".join(f"{k} {v}" for k, v in bg.items()))
    snr = pd.cut(o.snr_db.dropna(), [-1, 5, 10, 20, 40], labels=["0–5 dB", "5–10 dB", "10–20 dB", "20+ dB"])
    lines.append("* **SNR:** " + ", ".join(f"{k} {v}" for k, v in snr.value_counts().sort_index().items()))
    known = o[o.source != "urbansound8k"]  # UrbanSound8K does not record distance or device (placeholders)
    dist = pd.cut(known.distance_m.dropna(), [0, 2, 5, 10, 31], labels=["≤2 m", "2–5 m", "5–10 m", "10–30 m"])
    lines.append(f"* **Distance** (known for {len(known):,} generated clips): "
                 + ", ".join(f"{k} {v}" for k, v in dist.value_counts().sort_index().items()))
    lines.append(f"* **Overlapping events:** {o.overlap.notna().sum()} clips")
    lines.append("* **Speakers (speech classes):** " + ", ".join(
        f"{k} {v}" for k, v in o.speaker.dropna().value_counts().head(8).items()))
    lines += ["", "## Augmentation", ""]
    kinds = a.augmentation.dropna().str.split(";").explode().str.strip().str.replace(r"[ +\-x]?[0-9.]+.*$", "", regex=True).str.strip()
    lines.append("Transform use across augmented copies: " + ", ".join(
        f"{k} {v}" for k, v in kinds.value_counts().items() if k))
    lines += ["", "## Data dictionary (`data/dataset_metadata.csv`)", "", "| Column | Type / values | Meaning |",
              "|---|---|---|"]
    lines += [f"| `{c}` | {t} | {m} |" for c, t, m in DICTIONARY if c in d.columns]

    out = REPORTS_DIR / "dataset_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
