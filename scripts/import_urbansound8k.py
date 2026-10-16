"""Import UrbanSound8K dataset recordings into SonicSentinel incoming queue.

    python -m scripts.import_urbansound8k --zip <UrbanSound8K zip> --csv <UrbanSound8K.csv>
    python -m scripts.build_dataset

    # more clips of some classes, spread over many source recordings (at most 4 slices of one recording)
    python -m scripts.import_urbansound8k --only drilling jackhammer --count 200 --per-source 4

UrbanSound8K (CC-BY, https://urbansounddataset.weebly.com/urbansound8k.html) is
downloaded separately. Clips from one Freesound recording (``fsID``) share a
``source_group`` so they always land in the same split. UrbanSound8K does not
record the device or the distance, so those columns get the placeholders
``field_recorder`` and 5 m (see ``notes``).
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import pandas as pd

from config import DATASET_DIR

INCOMING = DATASET_DIR / "incoming"
ZIP_PATH = Path(r"C:\Users\tulip\Downloads\archive.zip")
CSV_PATH = Path(r"C:\Users\tulip\Downloads\UrbanSound8K.csv")

# Air conditioners and idling engines are steady ambient hum: Background Noise, not Machinery Fault
# (they were imported as Machinery Fault until 27 Sep 2026, see scripts/relabel_dataset.py).
# Counts raised on 28 Sep 2026: UrbanSound8K holds 8,732 real recordings and the project was
# using 3,922 of them. The everyday categories added that day (Engine Running, Household
# Appliance) and the animal species have real audio here, which the synthetic clips cannot match.
CLASS_MAPPING = {
    "gun_shot": ("gunshot", 374),
    "siren": ("alarm_or_siren", 500),
    "car_horn": ("vehicle_horn", 400),
    "dog_bark": ("dog_bark", 700),          # its own species class since 28 Sep 2026
    "drilling": ("drilling_or_grinder_sound", 300),
    "jackhammer": ("drilling_or_grinder_sound", 300),
    "engine_idling": ("engine_running", 600),   # Background Noise until 28 Sep 2026
    "air_conditioner": ("household_appliance", 500),  # Background Noise until 28 Sep 2026
    "street_music": ("background_noise", 300),
    "children_playing": ("background_noise", 300),
}


def select(rows: pd.DataFrame, count: int, per_source: int | None) -> pd.DataFrame:
    """The first ``count`` clips, or with ``per_source`` a round robin over the source recordings (fsID) that
    takes at most ``per_source`` slices of one recording, so the clips come from as many recordings as possible."""
    if not per_source:
        return rows.head(count)
    rank = rows.groupby("fsID").cumcount()
    return rows[rank < per_source].assign(_rank=rank).sort_values("_rank", kind="stable").head(count).drop(columns="_rank")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy the mapped UrbanSound8K classes into audio_dataset/incoming/.")
    parser.add_argument("--zip", type=Path, default=ZIP_PATH, help="UrbanSound8K archive (default: %(default)s)")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="UrbanSound8K.csv (default: %(default)s)")
    parser.add_argument("--only", nargs="*", help="UrbanSound8K classes to import (default: all mapped classes)")
    parser.add_argument("--count", type=int, help="clips per class (default: the count in CLASS_MAPPING)")
    parser.add_argument("--per-source", type=int, help="round robin over source recordings, at most N slices each")
    args = parser.parse_args(argv)
    if not args.zip.exists() or not args.csv.exists():
        print(f"Error: {args.zip} or {args.csv} not found.")
        return 1

    df = pd.read_csv(args.csv)
    meta_rows = []

    with zipfile.ZipFile(args.zip) as z:
        available = set(z.namelist())
        total_extracted = 0

        for u_class, (dest_slug, max_count) in CLASS_MAPPING.items():
            if args.only and u_class not in args.only:
                continue
            dest_dir = INCOMING / dest_slug
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            sub = select(df[df["class"] == u_class], args.count or max_count, args.per_source)
            extracted_for_class = 0

            for _, row in sub.iterrows():
                zip_member = f"fold{row.fold}/{row.slice_file_name}"
                if zip_member not in available:
                    continue
                
                target_file = dest_dir / row.slice_file_name
                if not target_file.exists():
                    data = z.read(zip_member)
                    target_file.write_bytes(data)
                
                meta_rows.append({
                    "filename": row.slice_file_name,
                    "environment": "urban",
                    "device": "field_recorder",
                    "distance_m": "5",
                    "source": "urbansound8k",
                    "license": "CC-BY",
                    "speaker": "",
                    "phrase": "",
                    "source_group": f"us8k_fsid_{row.fsID}",
                    "notes": f"UrbanSound8K fold{row.fold} {u_class}; device and distance unknown (placeholders)",
                })
                extracted_for_class += 1
                total_extracted += 1

            print(f"[{dest_slug}] Extracted {extracted_for_class} files for {u_class}")

    # Write incoming/metadata.csv
    meta_path = INCOMING / "metadata.csv"
    keys = ["filename", "environment", "device", "distance_m", "source", "license", "speaker", "phrase", "source_group", "notes"]
    if meta_path.exists():  # keep the rows of files imported earlier
        new = {r["filename"] for r in meta_rows}
        old = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
        meta_rows = [r for r in old.to_dict("records") if r.get("filename") not in new] + meta_rows
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"\nSuccessfully prepared {total_extracted} UrbanSound8K recordings in {INCOMING}!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
