"""Export Google Teachable Machine training samples from the *training* split.

    python -m gtm_model.export_gtm_samples [--per-recording 2]

GTM audio projects learn from ~1-second samples. For every training recording
(the same recordings the Python model was trained on) we:
  1. apply the same pre-processing as the web application
     (mono, 22.05 kHz, noise reduction, silence trim, peak normalisation),
  2. pick the loudest ~1 s window (and optionally a second, non-overlapping
     window within 10 dB of it) - the same rule the app uses at prediction time,
  3. up-sample to 44.1 kHz (GTM's sample rate) and save a 16-bit WAV.

Output (gtm_model/training_samples/):
  <Class Name>/<Audio ID>_w<k>.wav   individual samples (Audio ID kept for traceability)
  _concat/<Class Name>.wav           all samples of a class back to back, to play into
                                     GTM's recorder (see gtm_model/README.md)
  manifest.json                      counts and seconds per class
Validation and test recordings are never exported.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from audio_preprocessing import PreprocessConfig, decode_audio, preprocess_audio, write_wav
from audio_preprocessing.loader import resample
from config import DATASET_DIR, GTM_SAMPLES_DIR, METADATA_CSV, class_names, load_default_settings
from feature_extraction.gtm_spectrogram import GTM_INPUT_SAMPLES, GTM_SAMPLE_RATE

WINDOW_S = GTM_INPUT_SAMPLES / GTM_SAMPLE_RATE  # ~1.02 s


def windows_for(y: np.ndarray, sr: int, max_windows: int) -> list[np.ndarray]:
    win = int(round(WINDOW_S * sr))
    if len(y) <= win:
        pad = win - len(y)
        return [np.pad(y, (pad // 2, pad - pad // 2))]
    frame = max(1, int(0.05 * sr))
    n = len(y) // frame
    energy = np.square(y[: n * frame]).reshape(n, frame).sum(axis=1)
    order = np.argsort(-energy)
    out, taken = [], []
    for idx in order:
        centre = (idx + 0.5) * frame
        start = int(min(max(0, round(centre - win / 2)), len(y) - win))
        if any(abs(start - t) < win for t in taken):
            continue
        seg = y[start:start + win]
        if out and 10 * np.log10(np.mean(seg ** 2) + 1e-12) < 10 * np.log10(np.mean(out[0] ** 2) + 1e-12) - 10:
            break
        out.append(seg)
        taken.append(start)
        if len(out) >= max_windows:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-recording", type=int, default=2, help="max 1-second samples per training recording")
    parser.add_argument("--max-per-class", type=int, default=400)
    args = parser.parse_args(argv)

    meta = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    train = meta[(meta["split"] == "train") & (meta["status"] == "original")]
    pre_cfg = PreprocessConfig.from_settings(load_default_settings())
    if GTM_SAMPLES_DIR.exists():
        shutil.rmtree(GTM_SAMPLES_DIR)
    (GTM_SAMPLES_DIR / "_concat").mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "sample_rate": GTM_SAMPLE_RATE,
                "window_seconds": round(WINDOW_S, 4), "source_split": "train", "classes": {}}
    for cls in class_names():
        rows = train[train["class_label"] == cls]
        folder = GTM_SAMPLES_DIR / cls
        folder.mkdir(parents=True, exist_ok=True)
        concat, files = [], []
        for _, row in rows.iterrows():
            dec = decode_audio(DATASET_DIR / row["relative_path"])
            pre = preprocess_audio(dec.samples, dec.sample_rate, pre_cfg)
            for k, w in enumerate(windows_for(pre.y, pre.sr, args.per_recording)):
                w44 = resample(w, pre.sr, GTM_SAMPLE_RATE)[:GTM_INPUT_SAMPLES]
                if len(w44) < GTM_INPUT_SAMPLES:
                    w44 = np.pad(w44, (0, GTM_INPUT_SAMPLES - len(w44)))
                name = f"{row['audio_id']}_w{k + 1}.wav"
                write_wav(folder / name, w44, GTM_SAMPLE_RATE)
                files.append(name)
                concat.append(w44)
            if len(files) >= args.max_per_class:
                break
        if concat:
            write_wav(GTM_SAMPLES_DIR / "_concat" / f"{cls}.wav", np.concatenate(concat), GTM_SAMPLE_RATE)
        manifest["classes"][cls] = {"samples": len(files), "recordings": int(len(rows)),
                                    "seconds": round(len(files) * WINDOW_S, 1), "files": files[:5]}
        print(f"{cls:26s} {len(files):4d} samples from {len(rows)} training recordings")
    (GTM_SAMPLES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSamples written to {GTM_SAMPLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
