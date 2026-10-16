"""Move dataset recordings to another class (a label correction), keeping them traceable.

    python -m scripts.relabel_dataset --dry-run      # show what would change
    python -m scripts.relabel_dataset                # apply the corrections below

Corrections of 27 Sep 2026:
* UrbanSound8K ``air_conditioner`` and ``engine_idling`` had been imported as Machinery Fault. They are steady
  ambient hum - this project's Background Noise class covers HVAC and traffic hum - not machine faults, and
  they taught both models that any steady hum is a fault (an empty room with mains hum was classified as
  Machinery Fault). They move to Background Noise.
* UrbanSound8K ``drilling`` and ``jackhammer`` move from Machinery Fault to the optional category Drilling or
  Grinder Sound.
* Synthetic Background Noise clips that contain a door closing or knocking move to the optional category Door
  Impact, which now covers those sounds.

Corrections of 28 Sep 2026 (everyday sound categories):
* Background Noise clips generated purely as calm speech (``neutral-speech``, ``conversation``) move to
  Human Speech. They were the reason the live monitor answered "Background Noise" while someone was talking.
* Pure ``wind``, ``rain`` and ``everyday-clicks/keyboard`` / ``mouse-click`` beds move to Wind, Rain or Water
  and Keyboard or Mouse. Composite beds (``wind+conversation`` ...) stay Background Noise: mixed ambient with
  no single event is exactly what that category is for.

For every moved original recording and its augmented copies:
* a new Audio ID in the new class (``DS-BGN-<next number>``, copies ``<new ID>-AUG<k>``); old IDs are
  not reused;
* the file moves to ``<split>/<new class slug>/<new ID>.<ext>``; the split does not change;
* the old Audio ID and the reason are appended to ``notes``.
``data/dataset_metadata.csv``, ``audio_dataset/metadata.csv`` and ``data/dataset_statistics.json`` are
rewritten. Rebuild the features afterwards (``python -m feature_extraction.build_features``).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import PurePosixPath

from config import DATA_DIR, DATASET_DIR, class_code, slugify_class
from scripts.build_dataset import (
    _next_number,
    balance_report,
    dataset_statistics,
    load_metadata,
    print_table,
    save_metadata,
)

CORRECTIONS = [
    {
        "where": {"source": r"^urbansound8k$", "class_label": r"^Machinery Fault$",
                  "notes": r"\b(?:air_conditioner|engine_idling)$"},
        "to_class": "Background Noise",
        "reason": "steady ambient hum (air conditioner / idling engine) is background noise, not a machine fault",
    },
    {
        "where": {"source": r"^urbansound8k$", "class_label": r"^Machinery Fault$", "notes": r"\b(?:drilling|jackhammer)$"},
        "to_class": "Drilling or Grinder Sound",
        "reason": "power tools (drill, jackhammer) belong to the optional category Drilling or Grinder Sound",
    },
    {
        "where": {"class_label": r"^Background Noise$", "variant": r"door-close|knock"},
        "to_class": "Door Impact",
        "reason": "the clip contains a door closing or knocking, now the optional category Door Impact",
    },
    # Corrections of 28 Sep 2026, with the everyday sound categories. Only clips whose variant is a
    # *single* source move: a composite bed such as "wind+conversation" really is mixed ambient and stays
    # Background Noise.
    {
        "where": {"class_label": r"^Background Noise$", "variant": r"^(?:neutral-speech|conversation)$"},
        "to_class": "Human Speech",
        "reason": "the clip is a person talking calmly, which is now the category Human Speech",
    },
    {
        "where": {"class_label": r"^Background Noise$", "variant": r"^wind$"},
        "to_class": "Wind",
        "reason": "wind is now its own category",
    },
    {
        "where": {"class_label": r"^Background Noise$", "variant": r"^rain$"},
        "to_class": "Rain or Water",
        "reason": "rain is now part of the category Rain or Water",
    },
    {
        "where": {"class_label": r"^Background Noise$",
                  "variant": r"^everyday-clicks/(?:keyboard|mouse-click)$"},
        "to_class": "Keyboard or Mouse",
        "reason": "typing and mouse clicks are now the category Keyboard or Mouse",
    },
]
RELABEL_DATE = "2026-09-28"

# Animal species (28 Sep 2026). Every animal recording already names its animal: ESC-50 and
# UrbanSound8K in ``notes``, the procedurally generated clips in ``variant``. Animal Sound stays
# in the class list as the parent of these categories, so a detection is still an animal sound.
ANIMAL_SPECIES = [
    ("notes", r"\b(?:dog|dog_bark)$", "Dog Bark"),
    ("notes", r"\bcat$", "Cat"),
    ("notes", r"\b(?:chirping_birds|crow|rooster|hen)$", "Bird"),
    ("notes", r"\b(?:cow|pig|sheep)$", "Farm Animal"),
    ("notes", r"\b(?:insects|crickets|frog)$", "Insect or Frog"),
    ("variant", r"^dog-bark", "Dog Bark"),
    ("variant", r"^cat-meow$", "Cat"),
    ("variant", r"^(?:birds|crow|rooster|duck)$", "Bird"),
    ("variant", r"^(?:cow|sheep)$", "Farm Animal"),
    ("variant", r"^large-animal/", "Wild Animal"),
    ("variant", r"^frog$", "Insect or Frog"),
]
CORRECTIONS += [{
    "where": {"class_label": r"^Animal Sound$", column: pattern},
    "to_class": to_class,
    "reason": f"the recording is a {to_class.lower()}; Animal Sound is now their parent category",
} for column, pattern, to_class in ANIMAL_SPECIES]

# Back to the SRS category list (28 Sep 2026). Human Speech, the animal species and the everyday
# sounds were trialled and then removed: the SRS names ten mandatory categories plus seven optional
# ones, and nothing else. The recordings stay in the dataset - the species under Animal Sound, the
# everyday sounds under Background Noise - so the real audio imported from ESC-50 and UrbanSound8K
# still teaches the models what is, and what is not, an event.
TO_ANIMAL = ["Dog Bark", "Cat", "Bird", "Farm Animal", "Wild Animal", "Insect or Frog"]
TO_BACKGROUND = ["Human Speech", "Crying Baby", "Human Body Sound", "Laughter or Applause",
                 "Footsteps", "Keyboard or Mouse", "Household Appliance", "Rain or Water", "Wind",
                 "Thunderstorm", "Fire Crackling", "Church Bell", "Helicopter", "Train or Aircraft",
                 "Engine Running"]
CORRECTIONS += [{
    "where": {"class_label": rf"^{c}$"},
    "to_class": "Animal Sound",
    "reason": f"{c} was folded back into Animal Sound, the SRS category",
} for c in TO_ANIMAL]
CORRECTIONS += [{
    "where": {"class_label": rf"^{c}$"},
    "to_class": "Background Noise",
    "reason": f"{c} is not an SRS category; kept as Background Noise so the audio still trains the models",
} for c in TO_BACKGROUND]


def plan(df, corrections=CORRECTIONS) -> list[tuple[int, str, dict]]:
    """(row index, new Audio ID, correction) for every original recording to move, in Audio ID order."""
    moves = []
    counters: dict[str, int] = {}
    for c in corrections:
        mask = df["status"] == "original"
        for column, pattern in c["where"].items():
            mask &= df[column].str.contains(pattern, regex=True)
        hit = df[mask]
        code = class_code(c["to_class"])
        counters.setdefault(code, _next_number(df, code))
        for idx in hit.sort_values("audio_id").index:
            moves.append((idx, f"DS-{code}-{counters[code]:04d}", c))
            counters[code] += 1
    return moves


def _new_path(rel: str, old_id: str, new_id: str, new_class: str) -> str:
    p = PurePosixPath(rel)
    parts = list(p.parts)
    parts[-2] = slugify_class(new_class)
    parts[-1] = p.name.replace(old_id, new_id, 1)
    return str(PurePosixPath(*parts))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    df = load_metadata()
    moves = plan(df)
    if not moves:
        print("Nothing to relabel (already done).")
        return 0
    renames: list[tuple[str, str]] = []
    for idx, new_id, c in moves:
        old_id, old_class = df.at[idx, "audio_id"], df.at[idx, "class_label"]
        what = df.at[idx, "notes"] or df.at[idx, "variant"]
        copies = df.index[(df["root_audio_id"] == old_id) & (df.index != idx)]
        for j in [idx, *copies]:
            aid = df.at[j, "audio_id"]
            nid = aid.replace(old_id, new_id, 1)
            rel = df.at[j, "relative_path"]
            new_rel = _new_path(rel, old_id, new_id, c["to_class"])
            renames.append((rel, new_rel))
            note = f"relabelled from {aid} ({old_class}) on {RELABEL_DATE}: {c['reason']}"
            df.at[j, "notes"] = f"{df.at[j, 'notes']} | {note}" if df.at[j, "notes"] else note
            df.at[j, "audio_id"] = nid
            df.at[j, "root_audio_id"] = new_id
            if df.at[j, "parent_audio_id"]:
                df.at[j, "parent_audio_id"] = df.at[j, "parent_audio_id"].replace(old_id, new_id, 1)
            df.at[j, "class_label"] = c["to_class"]
            df.at[j, "filename"] = PurePosixPath(new_rel).name
            df.at[j, "relative_path"] = new_rel
        print(f"  {old_id} -> {new_id}  {df.at[idx, 'split']:10s} {what}  (+{len(copies)} augmented)")

    stats = dataset_statistics(df)
    stats["balance_warnings"] = balance_report(stats)
    print(f"\n{len(moves)} recordings relabelled ({len(renames)} files).")
    print_table(stats)
    if args.dry_run:
        return 0
    missing = [src for src, _ in renames if not (DATASET_DIR / src).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} files are missing, nothing changed (first: {missing[0]})")
    for src, dst in renames:
        (DATASET_DIR / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(DATASET_DIR / src), str(DATASET_DIR / dst))
    save_metadata(df)
    (DATA_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Metadata and statistics written. Next: python -m feature_extraction.build_features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
