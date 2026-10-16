"""Import real recordings from ESC-50 into the incoming queue.

    python -m scripts.import_esc50 --zip ESC-50-master.zip
    python -m scripts.build_dataset

ESC-50 (K. J. Piczak, https://github.com/karolpiczak/ESC-50) has 2,000 five-second recordings from
Freesound in 50 classes, 40 per class. The audio is licensed CC BY-NC 3.0 (attribution, non-commercial):
fine for this competition prototype, but a commercial deployment would need other data. Clips cut from the
same Freesound recording share a ``source_group`` (``esc50_<src_file>``), so they always land in the same
split.

Only classes that clearly belong to one of our categories are imported; everyday sounds with no event of
interest (rain, keyboard typing, coughing, vacuum cleaner ...) become Background Noise, which teaches the
models what is *not* an alert. Classes that fit nowhere (crying baby, helicopter, thunderstorm, hand saw ...)
are left out rather than forced into a wrong category.
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

from config import DATASET_DIR, slugify_class

INCOMING = DATASET_DIR / "incoming"
SOURCE = Path(r"C:/Users/tulip/Downloads/ESC-50")
BACKGROUND = "Background Noise"

CLASS_MAPPING = {
    # Animal Sound
    "dog": "Animal Sound", "rooster": "Animal Sound", "pig": "Animal Sound", "cow": "Animal Sound",
    "frog": "Animal Sound", "cat": "Animal Sound", "hen": "Animal Sound", "sheep": "Animal Sound",
    "crow": "Animal Sound", "chirping_birds": "Animal Sound", "insects": "Animal Sound", "crickets": "Animal Sound",
    # events of interest
    "glass_breaking": "Glass Breaking",
    "siren": "Alarm or Siren", "clock_alarm": "Alarm or Siren",
    "car_horn": "Vehicle Horn",
    "fireworks": "Fireworks",
    "door_wood_knock": "Door Impact",
    "chainsaw": "Drilling or Grinder Sound", "hand_saw": "Drilling or Grinder Sound",
    # everyday sound categories (added 28 Sep 2026). Before this they were all imported as Background
    # Noise, which is why the live monitor answered "Background Noise" for most real-world audio.
    "crying_baby": "Crying Baby",
    "coughing": "Human Body Sound", "sneezing": "Human Body Sound", "breathing": "Human Body Sound",
    "snoring": "Human Body Sound", "brushing_teeth": "Human Body Sound",
    "drinking_sipping": "Human Body Sound",
    "laughing": "Laughter or Applause", "clapping": "Laughter or Applause",
    "keyboard_typing": "Keyboard or Mouse", "mouse_click": "Keyboard or Mouse",
    "washing_machine": "Household Appliance", "vacuum_cleaner": "Household Appliance",
    "clock_tick": "Household Appliance", "can_opening": "Household Appliance",
    "rain": "Rain or Water", "water_drops": "Rain or Water", "pouring_water": "Rain or Water",
    "sea_waves": "Rain or Water", "toilet_flush": "Rain or Water",
    "wind": "Wind",
    "train": "Train or Aircraft", "airplane": "Train or Aircraft",
    "engine": "Engine Running",
    # nothing of interest here: a creaking door is ambient, and the five categories below were
    # dropped on 28 Sep 2026 (40 recordings each is too few to learn). Keeping them as Background
    # Noise still teaches the models that they are not events.
    "door_wood_creaks": "Background Noise", "footsteps": "Background Noise",
    "church_bells": "Background Noise", "crackling_fire": "Background Noise",
    "thunderstorm": "Background Noise", "helicopter": "Background Noise",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy mapped ESC-50 recordings into audio_dataset/incoming/.")
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="ESC-50 folder (with meta/ and audio/) or the downloaded master.zip")
    parser.add_argument("--background-per-class", type=int, default=16,
                        help="clips taken from each ESC-50 class that maps to Background Noise (40 = all); "
                             "keeps Background Noise from outgrowing the other categories")
    args = parser.parse_args(argv)
    if not args.source.exists():
        print(f"{args.source} not found. Download https://github.com/karolpiczak/ESC-50/archive/master.zip")
        return 1
    if args.source.is_dir():
        def read(rel: str) -> bytes:
            return (args.source / rel).read_bytes()
    else:
        z = zipfile.ZipFile(args.source)
        root = z.namelist()[0].split("/")[0]

        def read(rel: str) -> bytes:
            return z.read(f"{root}/{rel}")

    meta = pd.read_csv(io.BytesIO(read("meta/esc50.csv")))
    meta = meta[meta["category"].isin(CLASS_MAPPING)]
    # Background classes: a round robin over the source recordings, so the kept clips come from as many
    # different Freesound recordings as possible.
    keep = []
    for category, g in meta.groupby("category", sort=False):
        if CLASS_MAPPING[category] == BACKGROUND:
            rank = g.groupby("src_file").cumcount()
            g = g.assign(_r=rank).sort_values(["_r", "fold"], kind="stable").head(args.background_per_class)
        keep.append(g)
    meta = pd.concat(keep).drop(columns="_r", errors="ignore")

    rows = []
    for _, r in meta.iterrows():
        target = CLASS_MAPPING[r["category"]]
        dest = INCOMING / slugify_class(target) / f"esc50_{r['filename']}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(read(f"audio/{r['filename']}"))
        rows.append({"filename": dest.name, "environment": "", "device": "", "distance_m": "",
                     "source": "esc50", "license": "CC BY-NC 3.0 (ESC-50, Freesound)", "speaker": "",
                     "phrase": "", "source_group": f"esc50_{r['src_file']}",
                     "notes": f"ESC-50 fold{r['fold']} {r['category']}"})
    meta_path = INCOMING / "metadata.csv"
    keys = ["filename", "environment", "device", "distance_m", "source", "license", "speaker", "phrase",
            "source_group", "notes"]
    counts = pd.Series([CLASS_MAPPING[c] for c in meta["category"]]).value_counts()
    if meta_path.exists():
        new = {r["filename"] for r in rows}
        old = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
        rows = [r for r in old.to_dict("records") if r.get("filename") not in new] + rows
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(counts.to_string())
    print(f"{int(counts.sum())} ESC-50 recordings in {INCOMING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
