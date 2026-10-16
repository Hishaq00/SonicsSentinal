"""Build static/img/sonagram.png - the real mel spectrogram shown on the sign-in page.

    python -m scripts.make_brand_assets

It concatenates one test recording of several classes (siren, gunshot, a
"help" phrase, glass, a dog, a machinery fault) and renders their log-mel
spectrogram as ink on a dark panel. The image is data, not decoration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from audio_preprocessing.loader import load_mono
from config import DATASET_DIR, METADATA_CSV, PROJECT_ROOT

SR = 22050
ORDER = ["Alarm or Siren", "Gunshot", "Person Asking for Help", "Glass Breaking", "Animal Sound", "Machinery Fault"]
BG = np.array([27, 26, 23], dtype=float)
FG = np.array([239, 236, 229], dtype=float)


def main() -> None:
    import librosa

    meta = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    test = meta[(meta["split"] == "test") & (meta["status"] == "original")]
    parts = []
    for cls in ORDER:
        rows = test[test["class_label"] == cls]
        # Pick the clearest example: highest recorded SNR.
        snr = pd.to_numeric(rows["snr_db"], errors="coerce").fillna(0)
        row = rows.loc[snr.idxmax()]
        y, _ = load_mono(DATASET_DIR / row["relative_path"], SR)
        y = y / (np.max(np.abs(y)) + 1e-9)
        parts.append(y[: int(3.0 * SR)])
        parts.append(np.zeros(int(0.25 * SR), dtype=np.float32))
    y = np.concatenate(parts)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=192, n_mels=112, fmax=8000)
    db = librosa.power_to_db(mel, ref=np.max, top_db=70)
    v = np.clip((db + 70) / 70, 0, 1) ** 1.6
    rgb = (BG[None, None, :] + (FG - BG)[None, None, :] * v[::-1, :, None]).astype(np.uint8)
    img = Image.fromarray(rgb).resize((rgb.shape[1], 170), Image.Resampling.BICUBIC)
    out = PROJECT_ROOT / "static" / "img" / "sonagram.png"
    img.save(out, optimize=True)
    print(f"{out} ({img.width} x {img.height}, {len(y) / SR:.1f} s of audio)")


if __name__ == "__main__":
    main()
