"""Generate the synthetic bootstrap dataset.

Usage:
    python -m scripts.generate_synthetic_dataset                 # 300 clips per class
    python -m scripts.generate_synthetic_dataset --per-class 50  # quick test
    python -m scripts.generate_synthetic_dataset --variant empty-room --per-class 150 --start-index 300 --no-tts
                                                                 # extra Background Noise: quiet empty rooms
    (also --variant everyday-clicks / neutral-speech for Background Noise, large-animals for Animal Sound)

Output: ``audio_dataset/incoming/<class_slug>/*.wav`` plus
``audio_dataset/incoming/_synthetic_manifest.jsonl`` with the generation
parameters of every clip. Run ``python -m scripts.build_dataset`` afterwards
to assign Audio IDs, split the data and write the metadata CSV.

Why synthetic? The SRS requires each team to collect or create its own
dataset. These procedurally generated clips let the whole pipeline (training,
GTM export, the web app) run end-to-end on day one. They are labelled
``synthetic-*`` in the metadata and should be complemented or replaced with
real, ethically sourced recordings before the final evaluation.

Every clip varies: recording device, distance, indoor/outdoor environment,
sound intensity, background interference and SNR, echo/reverberation,
duration, clean vs noisy conditions, single vs overlapping events and
(for speech) different speakers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from audio_preprocessing.loader import write_wav
from config import DATASET_DIR, class_names, slugify_class
from scripts.synthetic import dsp
from scripts.synthetic.optional_sources import OPTIONAL_GENERATORS
from scripts.synthetic.sources import GENERATORS as BASE_GENERATORS
from scripts.synthetic.sources import VARIANT_GENERATORS, background_bed
from scripts.synthetic.tts import TTSLibrary

SR = dsp.SR
GENERATORS = {**BASE_GENERATORS, **OPTIONAL_GENERATORS}  # the ten mandatory + the SRS optional categories
INCOMING = DATASET_DIR / "incoming"
MANIFEST = INCOMING / "_synthetic_manifest.jsonl"

ENVS = list(dsp.ENVIRONMENTS)
CLASS_SPECS = {
    "Machinery Fault": {"dur": (2.5, 5.0), "env": [0.15, 0.15, 0.55, 0.15, 0.0]},
    "Glass Breaking": {"dur": (1.5, 4.0), "env": [0.35, 0.15, 0.0, 0.35, 0.15]},
    "Alarm or Siren": {"dur": (2.0, 5.0), "env": [0.2, 0.2, 0.0, 0.45, 0.15]},
    "Vehicle Horn": {"dur": (1.5, 4.0), "env": [0.0, 0.1, 0.0, 0.7, 0.2]},
    "Animal Sound": {"dur": (1.5, 4.5), "env": [0.2, 0.1, 0.0, 0.3, 0.4]},
    "Gunshot": {"dur": (1.5, 4.0), "env": [0.15, 0.15, 0.0, 0.4, 0.3]},
    "Panic Scream": {"dur": (1.5, 4.0), "env": [0.3, 0.2, 0.0, 0.3, 0.2]},
    "Aggression": {"dur": (2.0, 5.0), "env": [0.3, 0.25, 0.0, 0.35, 0.1]},
    "Person Asking for Help": {"dur": (1.5, 4.0), "env": [0.3, 0.2, 0.0, 0.3, 0.2]},
    "Background Noise": {"dur": (2.0, 5.0), "env": None},
    # SRS optional categories (environment weights: small room, hall, factory, street, open air)
    "Drone Sound": {"dur": (2.5, 5.0), "env": [0.05, 0.1, 0.0, 0.35, 0.5]},
    "Drilling or Grinder Sound": {"dur": (2.0, 5.0), "env": [0.25, 0.1, 0.4, 0.2, 0.05]},
    "Fireworks": {"dur": (2.5, 5.5), "env": [0.0, 0.0, 0.0, 0.4, 0.6]},
    "Vehicle Backfire": {"dur": (1.5, 4.0), "env": [0.0, 0.0, 0.0, 0.75, 0.25]},
    "Explosion": {"dur": (2.5, 5.5), "env": [0.05, 0.1, 0.15, 0.35, 0.35]},
    "Door Impact": {"dur": (1.5, 4.0), "env": [0.5, 0.3, 0.05, 0.15, 0.0]},
    "Graffiti Spray": {"dur": (2.0, 5.0), "env": [0.05, 0.1, 0.0, 0.6, 0.25]},
    # Everyday sound categories (28 Sep 2026). Only Human Speech is generated here, by the
    # neutral-speech variant; the others come from the real ESC-50 recordings.
    "Human Speech": {"dur": (2.0, 5.0), "env": [0.45, 0.25, 0.05, 0.2, 0.05]},
    "Wild Animal": {"dur": (1.5, 4.5), "env": [0.1, 0.05, 0.0, 0.25, 0.6]},
}
ENV_BEDS = {
    "indoor-small-room": ["room-tone", "hvac-office", "conversation"],
    "indoor-hall": ["crowd", "hvac-office", "room-tone"],
    "indoor-factory": ["machinery-normal", "hvac-office"],
    "outdoor-street": ["traffic", "street-mix", "wind", "rain"],
    "outdoor-open": ["wind", "rain", "room-tone"],
}
BED_TO_ENV = {
    "traffic": "outdoor-street", "street-mix": "outdoor-street", "rain": "outdoor-open",
    "wind": "outdoor-open", "hvac-office": "indoor-small-room", "room-tone": "indoor-small-room",
    "conversation": "indoor-small-room", "crowd": "indoor-hall", "machinery-normal": "indoor-factory",
    "empty-room": "indoor-small-room", "everyday-clicks": "indoor-small-room", "neutral-speech": "indoor-small-room",
}
QUIET_VARIANTS = {"empty-room"}  # recorded at a low level: nothing is happening
OVERLAP_CANDIDATES = {
    "Machinery Fault": ["Alarm or Siren", "Vehicle Horn"],
    "Glass Breaking": ["Animal Sound", "Vehicle Horn", "Alarm or Siren"],
    "Alarm or Siren": ["Vehicle Horn", "Animal Sound"],
    "Vehicle Horn": ["Animal Sound", "Alarm or Siren"],
    "Animal Sound": ["Vehicle Horn", "Machinery Fault"],
    "Gunshot": ["Vehicle Horn", "Animal Sound", "Machinery Fault"],
    "Panic Scream": ["Alarm or Siren", "Vehicle Horn", "Animal Sound"],
    "Aggression": ["Vehicle Horn", "Animal Sound", "Alarm or Siren"],
    "Person Asking for Help": ["Vehicle Horn", "Animal Sound", "Alarm or Siren"],
    "Drone Sound": ["Vehicle Horn", "Animal Sound"],
    "Drilling or Grinder Sound": ["Vehicle Horn", "Animal Sound"],
    "Fireworks": ["Animal Sound", "Vehicle Horn"],
    "Vehicle Backfire": ["Vehicle Horn", "Animal Sound"],
    "Explosion": ["Alarm or Siren", "Vehicle Horn"],
    "Door Impact": ["Animal Sound", "Vehicle Horn"],
    "Graffiti Spray": ["Vehicle Horn", "Animal Sound"],
}
LOUD_CLASSES = {"Gunshot", "Panic Scream", "Aggression", "Alarm or Siren", "Glass Breaking", "Vehicle Horn",
                "Explosion", "Fireworks", "Vehicle Backfire", "Door Impact", "Drilling or Grinder Sound"}


def seed_for(class_name: str, index: int, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{class_name}:{index}".encode()).hexdigest()
    return int(digest[:12], 16)


def _active_rms(x: np.ndarray) -> float:
    frame = 1024
    n = len(x) // frame
    if n < 2:
        return dsp.rms(x)
    e = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    act = e[e >= 0.1 * e.max()]
    return float(np.sqrt(np.mean(act ** 2))) if act.size else dsp.rms(x)


def build_clip(class_name: str, index: int, base_seed: int, tts: TTSLibrary | None,
               variant: str | None = None) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed_for(class_name, index, base_seed))
    spec = CLASS_SPECS[class_name]
    dur = float(rng.uniform(*spec["dur"]))
    n = dsp.n_samples(dur, SR)
    generator = VARIANT_GENERATORS[variant][1] if variant else GENERATORS[class_name]
    event, info = generator(rng, dur, SR, tts=tts)
    event = np.asarray(event, dtype=np.float64)

    canvas = np.zeros(n)
    if len(event) >= n:
        event = event[:n]
        start = 0
    else:
        margin = dsp.n_samples(0.05, SR)
        hi = max(margin + 1, n - len(event) - margin)
        start = int(rng.integers(margin, hi)) if hi > margin else 0
    dsp.place(canvas, event, start)

    overlap = None
    if class_name in OVERLAP_CANDIDATES and rng.random() < 0.12:
        overlap = OVERLAP_CANDIDATES[class_name][int(rng.integers(0, len(OVERLAP_CANDIDATES[class_name])))]
        sec, _ = GENERATORS[overlap](rng, dur * rng.uniform(0.4, 0.9), SR, tts=tts)
        rel_db = -float(rng.uniform(8, 16))
        sec = np.asarray(sec) * (_active_rms(event) / (_active_rms(sec) + 1e-9)) * dsp.db_to_gain(rel_db)
        dsp.place(canvas, sec, int(rng.integers(0, max(1, n - len(sec) // 2))))
        overlap = f"{overlap} ({rel_db:.0f} dB)"

    if class_name == "Background Noise":
        kind = info["variant"].split("+")[0].split("/")[0]
        env = BED_TO_ENV.get(kind, ENVS[int(rng.integers(0, len(ENVS)))])
    else:
        env = ENVS[int(rng.choice(len(ENVS), p=spec["env"]))]

    bucket = rng.choice(["near", "mid", "far"], p=[0.35, 0.4, 0.25])
    distance = float({"near": rng.uniform(0.5, 2.0), "mid": rng.uniform(2.0, 8.0), "far": rng.uniform(8.0, 30.0)}[bucket])
    y = dsp.apply_environment(canvas, env, distance, rng, SR)

    snr = None
    bed_kind = None
    if class_name != "Background Noise":
        bed_kind = ENV_BEDS[env][int(rng.integers(0, len(ENV_BEDS[env])))]
        bed, _ = background_bed(rng, n, SR, bed_kind, tts=tts)
        snr_bucket = rng.choice(["clean", "moderate", "noisy"], p=[0.4, 0.4, 0.2])
        snr = float({"clean": rng.uniform(20, 35), "moderate": rng.uniform(10, 20), "noisy": rng.uniform(0, 10)}[snr_bucket])
        y = dsp.mix_at_snr(y, bed, snr)

    device = list(dsp.DEVICES)[int(rng.integers(0, len(dsp.DEVICES)))]
    y = dsp.apply_device(y, device, rng, SR)

    # Sound intensity: distant sources are recorded quieter.
    level_db = float(np.clip(rng.uniform(-12, -2) - 10 * np.log10(max(distance, 1.0)), -35, -1))
    if variant in QUIET_VARIANTS:
        level_db = float(rng.uniform(-40, -18))
    y = dsp.peak_normalize(y, dsp.db_to_gain(level_db))
    clipped = False
    if class_name in LOUD_CLASSES and rng.random() < 0.06:
        y = np.clip(dsp.peak_normalize(y, rng.uniform(1.2, 2.0)), -0.999, 0.999)
        clipped = True
        level_db = 0.0
    y = dsp.fade(y, 0.005, 0.01, SR).astype(np.float32)

    meta = {
        "class_label": class_name,
        "source": "synthetic-tts" if "tts" in str(info.get("variant", "")) else "synthetic-procedural",
        "license": "CC0 (generated by the project team with scripts/generate_synthetic_dataset.py)",
        "variant": info.get("variant", ""),
        "environment": env,
        "device": device,
        "distance_m": round(distance, 1),
        "distance_bucket": str(bucket),
        "background": bed_kind or info.get("variant", ""),
        "snr_db": None if snr is None else round(snr, 1),
        "overlap": overlap or "",
        "peak_level_dbfs": round(level_db, 1),
        "clipped": clipped,
        "speaker": info.get("speaker", ""),
        "phrase": " | ".join(info.get("phrases", [])),
        "source_group": info.get("source_group", ""),
        "seed": seed_for(class_name, index, base_seed),
    }
    return y, meta


def _generate_one(class_name: str, index: int, base_seed: int, tts: TTSLibrary | None,
                  variant: str | None = None) -> dict:
    y, meta = build_clip(class_name, index, base_seed, tts, variant)
    slug = slugify_class(class_name)
    fname = f"syn_{slug}_{index:04d}.wav"
    path = INCOMING / slug / fname
    write_wav(path, y, SR)
    meta["filename"] = fname
    meta["relative_path"] = str(path.relative_to(DATASET_DIR)).replace("\\", "/")
    meta["duration_s"] = round(len(y) / SR, 3)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jobs", type=int, default=-1, help="parallel worker processes (-1 = all cores)")
    parser.add_argument("--classes", nargs="*", help="subset of class names")
    parser.add_argument("--no-tts", action="store_true", help="do not use text-to-speech voices")
    parser.add_argument("--variant", choices=sorted(VARIANT_GENERATORS),
                        help="generate only this extra variant (its class is implied)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="first clip number; use one above the existing clips so new files get new seeds")
    args = parser.parse_args(argv)

    classes = [VARIANT_GENERATORS[args.variant][0]] if args.variant else (args.classes or class_names())
    # A --variant run brings its own generator, so its class does not need one of its own
    # (Human Speech, for example, is generated only by the neutral-speech variant).
    if args.variant:
        pass  # the variant brings its own generator
    elif args.classes:
        unknown = [c for c in classes if c not in GENERATORS]
        if unknown:
            print(f"No generator for: {unknown}", file=sys.stderr)
            return 2
    else:
        # Whole-dataset run: generate the categories that have a procedural generator. The rest
        # (the everyday sounds and most animal species) come from real recordings - see
        # scripts/import_esc50.py and scripts/import_urbansound8k.py.
        skipped = [c for c in classes if c not in GENERATORS]
        classes = [c for c in classes if c in GENERATORS]
        if skipped:
            print(f"No procedural generator (import real recordings instead): {', '.join(skipped)}")
    tts = None if args.no_tts else TTSLibrary.build(DATASET_DIR / "_tts_cache")
    INCOMING.mkdir(parents=True, exist_ok=True)

    jobs = [(c, i) for c in classes for i in range(args.start_index, args.start_index + args.per_class)]
    print(f"Generating {len(jobs)} clips ({args.per_class} per class{', ' + args.variant if args.variant else ''}) ...")
    t0 = time.time()
    from joblib import Parallel, delayed

    metas = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_generate_one)(c, i, args.seed, tts, args.variant) for c, i in jobs)
    # Merge with an existing manifest (other classes generated earlier).
    existing = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                existing[m["relative_path"]] = m
    for m in metas:
        existing[m["relative_path"]] = m
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        for m in existing.values():
            fh.write(json.dumps(m) + "\n")
    print(f"Done in {time.time() - t0:.0f} s. Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
