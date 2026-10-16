"""Build / update the common sound dataset.

    python -m scripts.build_dataset

What it does
1. Scans ``audio_dataset/incoming/<class>/`` for new recordings (WAV, MP3,
   FLAC, OGG, M4A). The folder name is the class (class name or slug).
2. Validates every file (decodable, not silent, not too short) and skips exact
   duplicates (SHA-256 of the file bytes).
3. Assigns a unique Audio ID: ``DS-<CLASS CODE>-<NNNN>`` (e.g. ``DS-GUN-0042``).
4. Assigns a stratified 70 / 15 / 15 train / validation / test split per class.
   Recordings that share a ``source_group`` (e.g. the same text-to-speech
   phrase recording) always land in the same split, so no source leaks from
   training into testing.
5. Moves the file to ``audio_dataset/<split>/<class_slug>/<Audio ID>.<ext>``.
6. Writes ``data/dataset_metadata.csv`` (also copied to
   ``audio_dataset/metadata.csv``) and ``data/dataset_statistics.json``, and
   prints a class-balance report.

Extra metadata for real recordings can be supplied in
``audio_dataset/incoming/metadata.csv`` with the columns
``filename, environment, device, distance_m, source, license, speaker, phrase, source_group, notes``.
Existing Audio IDs and splits are never changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audio_preprocessing.loader import SUPPORTED_EXTENSIONS, AudioDecodeError, decode_audio, sha256_file
from audio_preprocessing.quality import analyze_quality
from audio_preprocessing.validation import loudest_frame_dbfs
from config import (
    DATA_DIR,
    DATASET_DIR,
    METADATA_CSV,
    class_code,
    class_from_folder,
    class_names,
    slugify_class,
)

INCOMING = DATASET_DIR / "incoming"
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
TARGET_PER_CLASS = 300

COLUMNS = [
    "audio_id", "root_audio_id", "parent_audio_id", "status", "class_label", "split",
    "filename", "original_filename", "relative_path", "source", "license", "source_group",
    "duration_s", "sample_rate", "channels", "bit_depth", "format", "file_size_bytes", "sha256",
    "environment", "device", "distance_m", "background", "snr_db", "overlap", "speaker", "phrase",
    "variant", "quality_label", "quality_score", "augmentation", "created_at", "notes",
]


def load_metadata(path: Path = METADATA_CSV) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def save_metadata(df: pd.DataFrame, path: Path = METADATA_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df[COLUMNS].sort_values(["class_label", "audio_id"])
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    shutil.copyfile(path, DATASET_DIR / "metadata.csv")


def _synthetic_manifest() -> dict[str, dict]:
    manifest = INCOMING / "_synthetic_manifest.jsonl"
    out: dict[str, dict] = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                out[m["filename"]] = m
    return out


def _team_metadata() -> dict[str, dict]:
    path = INCOMING / "metadata.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {row["filename"]: row.to_dict() for _, row in df.iterrows() if row.get("filename")}


def _next_number(df: pd.DataFrame, code: str) -> int:
    prefix = f"DS-{code}-"
    nums = [int(a[len(prefix):len(prefix) + 4]) for a in df["audio_id"] if a.startswith(prefix) and a[len(prefix):len(prefix) + 4].isdigit()]
    return (max(nums) + 1) if nums else 1


def assign_splits(new_rows: list[dict], existing: pd.DataFrame, seed: int) -> None:
    """Greedy stratified group split that tops up each split towards 70/15/15."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in new_rows:
        by_class[row["class_label"]][row["source_group"] or row["audio_id"]].append(row)
    orig = existing[existing["status"] == "original"] if len(existing) else existing
    # A source group already in the dataset keeps the split it was given, even when the new clips
    # belong to another class: the same text-to-speech phrase recording must never be trained on in
    # one split and tested on in another.
    placed: dict[str, str] = {}
    if len(orig):
        for group, split in zip(orig["source_group"], orig["split"]):
            if group:
                placed.setdefault(group, split)
    for cls, groups in by_class.items():
        counts = Counter(orig[orig["class_label"] == cls]["split"]) if len(orig) else Counter()
        keys = list(groups)
        rng.shuffle(keys)
        # Bigger groups first so they do not unbalance a nearly-full split at the end.
        keys.sort(key=lambda k: -len(groups[k]))
        for key in keys:
            if key in placed:
                split = placed[key]
            else:
                total = sum(counts.values()) + len(groups[key])
                deficits = {s: RATIOS[s] * total - counts[s] for s in SPLITS}
                split = max(SPLITS, key=lambda s: deficits[s])
            for row in groups[key]:
                row["split"] = split
            counts[split] += len(groups[key])


def scan_incoming(df: pd.DataFrame, seed: int, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    report = {"added": 0, "duplicates": [], "rejected": [], "unknown_class_folders": []}
    if not INCOMING.exists():
        return df, report
    synthetic = _synthetic_manifest()
    team = _team_metadata()
    known_hashes = set(df["sha256"]) if len(df) else set()
    counters: dict[str, int] = {}
    new_rows: list[dict] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for folder in sorted(p for p in INCOMING.iterdir() if p.is_dir() and not p.name.startswith("_")):
        cls = class_from_folder(folder.name)
        if cls is None:
            report["unknown_class_folders"].append(folder.name)
            continue
        code = class_code(cls)
        for path in sorted(folder.iterdir()):
            if path.suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
                continue
            digest = sha256_file(path)
            if digest in known_hashes:
                report["duplicates"].append(str(path.relative_to(DATASET_DIR)))
                continue
            try:
                dec = decode_audio(path)
            except AudioDecodeError as exc:
                report["rejected"].append({"file": path.name, "reason": str(exc)})
                continue
            mono = dec.mono()
            if dec.duration < 0.3:
                report["rejected"].append({"file": path.name, "reason": f"too short ({dec.duration:.2f} s)"})
                continue
            if loudest_frame_dbfs(mono, dec.sample_rate) < -60:
                report["rejected"].append({"file": path.name, "reason": "silent"})
                continue
            known_hashes.add(digest)
            if code not in counters:
                counters[code] = _next_number(df, code)
            audio_id = f"DS-{code}-{counters[code]:04d}"
            counters[code] += 1
            quality = analyze_quality(mono, dec.sample_rate, bit_depth=dec.bit_depth, decode_warnings=dec.warnings)
            extra = synthetic.get(path.name) or team.get(path.name) or {}
            row = {c: "" for c in COLUMNS}
            row.update({
                "audio_id": audio_id,
                "root_audio_id": audio_id,
                "status": "original",
                "class_label": cls,
                "filename": f"{audio_id}{path.suffix.lower()}",
                "original_filename": path.name,
                "source": extra.get("source", "recorded"),
                "license": extra.get("license", ""),
                "source_group": extra.get("source_group", "") or "",
                "duration_s": f"{dec.duration:.3f}",
                "sample_rate": str(dec.sample_rate),
                "channels": str(dec.channels),
                "bit_depth": "" if dec.bit_depth is None else str(dec.bit_depth),
                "format": dec.format,
                "file_size_bytes": str(path.stat().st_size),
                "sha256": digest,
                "environment": extra.get("environment", ""),
                "device": extra.get("device", ""),
                "distance_m": str(extra.get("distance_m", "") or ""),
                "background": extra.get("background", ""),
                "snr_db": "" if extra.get("snr_db") in (None, "") else str(extra.get("snr_db")),
                "overlap": extra.get("overlap", ""),
                "speaker": extra.get("speaker", ""),
                "phrase": extra.get("phrase", ""),
                "variant": extra.get("variant", ""),
                "quality_label": quality["label"],
                "quality_score": str(quality["score"]),
                "created_at": now,
                "notes": extra.get("notes", ""),
                "_path": path,
            })
            new_rows.append(row)

    assign_splits(new_rows, df, seed)
    for row in new_rows:
        src: Path = row.pop("_path")
        dest = DATASET_DIR / row["split"] / slugify_class(row["class_label"]) / row["filename"]
        row["relative_path"] = str(dest.relative_to(DATASET_DIR)).replace("\\", "/")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
    report["added"] = len(new_rows)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows, columns=COLUMNS)], ignore_index=True)
    return df, report


def dataset_statistics(df: pd.DataFrame) -> dict:
    stats: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "classes": {}}
    orig = df[df["status"] == "original"]
    aug = df[df["status"] == "augmented"]
    stats["total_original"] = int(len(orig))
    stats["total_augmented"] = int(len(aug))
    stats["split_totals"] = {s: int((orig["split"] == s).sum()) for s in SPLITS}
    durations = pd.to_numeric(orig["duration_s"], errors="coerce")
    stats["total_hours"] = round(float(durations.sum()) / 3600.0, 2)
    for cls in class_names():
        c = orig[orig["class_label"] == cls]
        d = pd.to_numeric(c["duration_s"], errors="coerce")
        stats["classes"][cls] = {
            "original": int(len(c)),
            "augmented": int((aug["class_label"] == cls).sum()),
            "splits": {s: int((c["split"] == s).sum()) for s in SPLITS},
            "mean_duration_s": round(float(d.mean()), 2) if len(c) else 0.0,
            "sources": dict(Counter(c["source"])),
            "environments": dict(Counter(e for e in c["environment"] if e)),
            "devices": dict(Counter(e for e in c["device"] if e)),
            "quality": dict(Counter(c["quality_label"])),
        }
    return stats


def balance_report(stats: dict) -> list[str]:
    """Class-imbalance and insufficient-sample warnings (FR xviii)."""
    warnings: list[str] = []
    counts = {c: v["original"] for c, v in stats["classes"].items()}
    if not counts:
        return ["Dataset is empty."]
    mean = np.mean(list(counts.values()))
    for cls, n in counts.items():
        if n == 0:
            warnings.append(f"{cls}: no recordings.")
        elif n < TARGET_PER_CLASS:
            warnings.append(f"{cls}: only {n} original clips (target {TARGET_PER_CLASS}).")
        if mean and abs(n - mean) / mean > 0.10:
            warnings.append(f"{cls}: {n} clips is more than 10% away from the class mean ({mean:.0f}).")
        splits = stats["classes"][cls]["splits"]
        if n and splits.get("test", 0) < 10:
            warnings.append(f"{cls}: fewer than 10 test recordings ({splits.get('test', 0)}).")
    return warnings


def print_table(stats: dict) -> None:
    print(f"\n{'Class':26s} {'train':>6s} {'val':>6s} {'test':>6s} {'total':>6s} {'aug':>6s}")
    for cls, v in stats["classes"].items():
        s = v["splits"]
        print(f"{cls:26s} {s['train']:6d} {s['validation']:6d} {s['test']:6d} {v['original']:6d} {v['augmented']:6d}")
    t = stats["split_totals"]
    print(f"{'TOTAL':26s} {t['train']:6d} {t['validation']:6d} {t['test']:6d} {stats['total_original']:6d} {stats['total_augmented']:6d}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drop-missing", action="store_true", help="remove rows whose file no longer exists")
    args = parser.parse_args(argv)

    df = load_metadata()
    if args.drop_missing and len(df):
        exists = df["relative_path"].map(lambda p: (DATASET_DIR / p).exists())
        dropped = int((~exists).sum())
        df = df[exists].reset_index(drop=True)
        print(f"Dropped {dropped} rows with missing files.")
    df, report = scan_incoming(df, args.seed, args.dry_run)
    print(f"Added {report['added']} recordings; {len(report['duplicates'])} duplicates skipped; "
          f"{len(report['rejected'])} rejected.")
    for r in report["rejected"][:20]:
        print(f"  rejected {r['file']}: {r['reason']}")
    if report["unknown_class_folders"]:
        print(f"  unknown class folders: {report['unknown_class_folders']}")
    stats = dataset_statistics(df)
    stats["balance_warnings"] = balance_report(stats)
    print_table(stats)
    for w in stats["balance_warnings"]:
        print("  WARNING:", w)
    if not args.dry_run:
        save_metadata(df)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\nMetadata written to {METADATA_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
