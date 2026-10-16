"""Pick one example recording per class for ``sample_audio/`` (files evaluators can upload to try the app).

    python -m scripts.select_sample_audio

For every class the candidates are original *test-split* recordings of 1.5-6 s. Each is analysed like an
upload (pre-processing, 2 s segments, mean over the event segments) by the active Python model and the GTM
model in ``gtm_model/model/``. The example is the recording both models assign to its class with the highest
confidence (the lower of the two); real recordings (UrbanSound8K) are preferred over generated ones when
both models get one right. These are clear examples for a demonstration, not a measure of accuracy - the
accuracy is in ``reports/``.

Writes ``sample_audio/sample_<class slug>.wav`` and ``sample_audio/README.md``.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from audio_preprocessing import decode_audio
from config import DATASET_DIR, METADATA_CSV, PROJECT_ROOT, class_info, class_names, load_default_settings
from gtm_model import pipeline as gp
from gtm_model.inference import GTMClassifier
from python_models.inference import get_classifier

OUT = PROJECT_ROOT / "sample_audio"
CANDIDATES_PER_CLASS = 25


def recording_scores(path, clf, gtm, settings) -> tuple[dict, dict]:
    dec = decode_audio(path)
    audio = gp.upload_audio(dec.samples, dec.sample_rate, settings)
    segs = gp.app_segments(audio)
    events = [s for s, e in segs if e] or [s for s, _ in segs]
    py = clf.predict_segments([s.audio for s in events])
    gt = gtm.predict_inputs([gp.segment_input(audio, s)[0] for s in events])
    classes = clf.classes
    return ({c: float(np.mean([p[c] for p in py])) for c in classes},
            {c: float(np.mean([p[c] for p in gt])) for c in classes})


def main() -> int:
    settings = load_default_settings()
    clf, gtm = get_classifier(), GTMClassifier()
    meta = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    dur = pd.to_numeric(meta["duration_s"], errors="coerce")
    test = meta[(meta["split"] == "test") & (meta["status"] == "original") & dur.between(1.5, 6.0)]
    OUT.mkdir(exist_ok=True)
    rows = []
    for cls in class_names():
        cand = test[test["class_label"] == cls]
        real = cand[cand["source"] == "urbansound8k"].head(CANDIDATES_PER_CLASS)
        generated = cand[cand["source"] != "urbansound8k"].head(CANDIDATES_PER_CLASS)
        best = None
        for group in (real, generated):
            for _, r in group.iterrows():
                py, gt = recording_scores(DATASET_DIR / r["relative_path"], clf, gtm, settings)
                if max(py, key=py.get) != cls or max(gt, key=gt.get) != cls:
                    continue
                score = min(py[cls], gt[cls])
                if best is None or score > best[0]:
                    best = (score, r, py[cls], gt[cls])
            if best is not None:
                break
        if best is None:
            print(f"{cls}: no test recording that both models classify correctly")
            continue
        _, r, p, g = best
        dest = OUT / f"sample_{class_info()[cls]['slug']}{(DATASET_DIR / r['relative_path']).suffix}"
        shutil.copyfile(DATASET_DIR / r["relative_path"], dest)
        rows.append({"class": cls, "file": dest.name, "audio_id": r["audio_id"], "source": r["source"],
                     "variant": r["variant"] or r["notes"].split("|")[0].strip(), "duration": r["duration_s"],
                     "python": p, "gtm": g})
        print(f"{cls:26s} {dest.name:40s} {r['audio_id']}  Python {p:.2f}  GTM {g:.2f}")

    lines = [
        "# Sample audio", "",
        "One example recording per sound category for trying the app: log in, open **Analyze audio** (`/analyze`) "
        "and upload a file.", "",
        f"Chosen by `python -m scripts.select_sample_audio` from the **test split** (recordings no model was trained "
        f"on): for each class, the recording that both models (Python `{clf.version}`, GTM `{gtm.version}`) "
        "classify correctly with the highest confidence, preferring real UrbanSound8K recordings. They are clear "
        "examples for a demonstration; the models are not right on every recording - see `reports/` for the "
        "measured accuracy.", "",
        "| Category | File | Audio ID | Source | Duration (s) | Python | GTM |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['class']} | `{r['file']}` | {r['audio_id']} | {r['source']} ({r['variant']}) | "
                     f"{float(r['duration']):.1f} | {r['python'] * 100:.0f}% | {r['gtm'] * 100:.0f}% |")
    lines += ["", "Percentages are the confidence each model gives the correct category when the file is analysed "
              "like an upload (mean over the event segments)."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
