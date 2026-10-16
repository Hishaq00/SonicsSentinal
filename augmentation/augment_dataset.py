"""Create augmented copies of the *training* recordings.

    python -m augmentation.augment_dataset --copies 1      # one copy of every training recording
    python -m augmentation.augment_dataset --balance 420   # level the classes out at ~420 training rows

* Only rows with ``split == train`` and ``status == original`` are augmented.
* ``--balance N`` chooses the number of copies per class, so a category with few
  recordings (the everyday sounds from ESC-50 have 40 each) is not drowned out by
  one with thousands.
* Each copy gets 2-3 random transforms (noise, time shift, pitch shift, time
  stretch, volume, limited reverberation, distance or device simulation).
* The background-noise bank uses training-split Background Noise recordings
  only, so validation / test audio never leaks into training.
* Augmented rows keep the parent's split and class, get the ID
  ``<parent Audio ID>-AUG<k>`` and ``status = augmented``. They are never
  counted as unique original clips.
* Re-running replaces the previous augmented set (idempotent).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from audio_preprocessing.loader import load_mono, write_wav
from augmentation.transforms import random_chain
from config import DATASET_DIR, slugify_class
from scripts.build_dataset import COLUMNS, dataset_statistics, load_metadata, print_table, save_metadata

SR = 22050
AUG_DIR = DATASET_DIR / "augmented"


def _seed(audio_id: str, k: int, base: int) -> int:
    return int(hashlib.sha256(f"{base}:{audio_id}:{k}".encode()).hexdigest()[:12], 16)


def _augment_one(row: dict, k: int, base_seed: int, noise_bank: list[np.ndarray]) -> dict:
    rng = np.random.default_rng(_seed(row["audio_id"], k, base_seed))
    y, sr = load_mono(DATASET_DIR / row["relative_path"], SR)
    out, applied = random_chain(y, sr, rng, noise_bank)
    new_id = f"{row['audio_id']}-AUG{k}"
    dest = AUG_DIR / "train" / slugify_class(row["class_label"]) / f"{new_id}.wav"
    write_wav(dest, out, sr)
    new = {c: "" for c in COLUMNS}
    new.update({
        "audio_id": new_id,
        "root_audio_id": row["root_audio_id"] or row["audio_id"],
        "parent_audio_id": row["audio_id"],
        "status": "augmented",
        "class_label": row["class_label"],
        "split": row["split"],
        "filename": dest.name,
        "original_filename": row["filename"],
        "relative_path": str(dest.relative_to(DATASET_DIR)).replace("\\", "/"),
        "source": row["source"],
        "license": row["license"],
        "source_group": row["source_group"],
        "duration_s": f"{len(out) / sr:.3f}",
        "sample_rate": str(sr),
        "channels": "1",
        "bit_depth": "16",
        "format": "WAV",
        "file_size_bytes": str(dest.stat().st_size),
        "environment": row["environment"],
        "device": row["device"],
        "distance_m": row["distance_m"],
        "augmentation": "; ".join(applied),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copies", type=int, default=1, help="augmented copies per training recording")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--balance", type=int, metavar="N",
                        help="instead of a fixed number of copies, give every class about N training rows: "
                             "a class with few recordings gets more copies (at most --max-copies). Classes "
                             "that already have N or more get none.")
    parser.add_argument("--max-copies", type=int, default=12,
                        help="upper limit per recording when --balance is used")
    args = parser.parse_args(argv)

    df = load_metadata()
    if not len(df):
        print("No dataset metadata found. Run: python -m scripts.build_dataset")
        return 1
    df = df[df["status"] != "augmented"].reset_index(drop=True)
    if AUG_DIR.exists():
        shutil.rmtree(AUG_DIR)

    train = df[(df["split"] == "train") & (df["status"] == "original")]
    bank_rows = train[train["class_label"] == "Background Noise"].head(80)
    noise_bank = [load_mono(DATASET_DIR / p, SR)[0] for p in bank_rows["relative_path"]]

    per_class = train["class_label"].value_counts()
    if args.balance:
        copies = {cls: min(args.max_copies, max(0, -(-args.balance // n) - 1)) for cls, n in per_class.items()}
        print(f"Augmenting to about {args.balance} training rows per class "
              f"(noise bank: {len(noise_bank)} background recordings) ...")
        for cls in sorted(copies, key=lambda c: -copies[c]):
            print(f"  {cls:26s} {per_class[cls]:5d} recordings x {copies[cls]} copies "
                  f"-> {per_class[cls] * (copies[cls] + 1):5d} rows")
    else:
        copies = {cls: args.copies for cls in per_class.index}
        print(f"Augmenting {len(train)} training recordings x {args.copies} copies "
              f"(noise bank: {len(noise_bank)} background recordings) ...")

    from joblib import Parallel, delayed

    rows = train.to_dict("records")
    new_rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_augment_one)(r, k, args.seed, noise_bank)
        for r in rows for k in range(1, copies.get(r["class_label"], 0) + 1)
    )
    df = pd.concat([df, pd.DataFrame(new_rows, columns=COLUMNS)], ignore_index=True)
    save_metadata(df)
    stats = dataset_statistics(df)
    print_table(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
