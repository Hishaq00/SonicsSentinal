# audio_dataset — the common sound dataset

One dataset is used by **both** models (SRS step 5): the Python model trains on features of these recordings,
the GTM model on ~1 s samples cut from the same training recordings.

| | |
|---|---|
| Classes | 17: the 10 mandatory SRS categories + the 7 SRS optional categories (`config/classes.json`) |
| Unique original clips | 12,018 (≥ 300 per class; largest class 3,644) |
| Augmented copies | 5,038 (training split only) |
| Split | train 8,405 · validation 1,810 · test 1,803 original clips (stratified 70 / 15 / 15); train becomes 13,443 once the augmented copies are added |
| Total metadata rows | 17,056 |
| Size | ~1.9 GB (not in git — see "Getting the audio") |
| Metadata | `metadata.csv` (this folder) = `data/dataset_metadata.csv` |
| Statistics, quality report, data dictionary | `reports/dataset_report.md` (`python -m scripts.dataset_report`) |

## Layout

```
audio_dataset/
  train/<class_slug>/<Audio ID>.wav        originals, training split
  validation/<class_slug>/<Audio ID>.wav
  test/<class_slug>/<Audio ID>.wav
  augmented/<class_slug>/<Audio ID>-AUG1.wav   copies of training recordings only
  incoming/                                  drop new recordings here (one folder per class)
  metadata.csv
```

Audio IDs: `DS-<class code>-<NNNN>`, e.g. `DS-GUN-0042`; augmented copies `DS-GUN-0042-AUG1` keep the root
Audio ID and split. Segments cut from a recording for training keep its Audio ID and class.

## Getting the audio

The audio folders are too large for GitHub. Either

* copy it out of the submission — the same audio ships in `04 - Training Dataset/train/`,
  `05 - Validation Dataset/validation/` and `06 - Testing Dataset/test/`. Copy those three folders in here as
  `train/`, `validation/` and `test/`, then run `python -m augmentation.augment_dataset --copies 1` to
  recreate `augmented/`, or
* rebuild it:

```bash
python -m scripts.generate_synthetic_dataset                 # 300 procedural / TTS clips per class -> incoming/
python -m scripts.import_urbansound8k --zip UrbanSound8K.zip --csv UrbanSound8K.csv   # real recordings -> incoming/
python -m scripts.build_dataset                              # Audio IDs, validation, duplicates, split, metadata
python -m scripts.relabel_dataset                            # label corrections of 27 Sep 2026 (see below)
python -m scripts.import_urbansound8k --only drilling --count 260 --per-source 4
python -m scripts.import_urbansound8k --only jackhammer --count 160 --per-source 4
python -m scripts.generate_synthetic_dataset --classes "Drone Sound" Fireworks "Vehicle Backfire" Explosion "Door Impact" "Graffiti Spray" --per-class 300
python -m scripts.generate_synthetic_dataset --classes "Drilling or Grinder Sound" --per-class 150
python -m scripts.generate_synthetic_dataset --variant empty-room --per-class 150 --start-index 300 --no-tts
python -m scripts.generate_synthetic_dataset --variant everyday-clicks --per-class 150 --start-index 450
python -m scripts.generate_synthetic_dataset --variant neutral-speech --per-class 150 --start-index 600
python -m scripts.generate_synthetic_dataset --variant large-animals --per-class 150 --start-index 300 --no-tts
python -m scripts.build_dataset
python -m augmentation.augment_dataset --copies 1            # augmented training copies
python -m feature_extraction.build_features                  # features for the Python model
```

UrbanSound8K is published under CC-BY by its authors and must be downloaded from its website. Generated audio is
CC0. Procedural clips are deterministic for a given seed; text-to-speech clips use the Windows SAPI voices
installed on the machine, so a rebuild on another computer sounds different — the published archive is the
reference copy.

## Changes of 27 Sep 2026

* **Label corrections** (`scripts/relabel_dataset.py`, old Audio IDs kept in `notes`): UrbanSound8K
  `air_conditioner` and `engine_idling` moved from Machinery Fault to Background Noise (steady ambient hum is
  not a fault; they taught the models that an empty room with mains hum is a machine fault); UrbanSound8K
  `drilling` and `jackhammer` moved to the new category Drilling or Grinder Sound; synthetic Background Noise
  clips containing a door closing or knocking moved to Door Impact.
* **SRS optional categories** added: Drone Sound, Drilling or Grinder Sound (UrbanSound8K drilling and
  jackhammer from many source recordings + generated drills and grinders), Fireworks, Vehicle Backfire,
  Explosion, Door Impact and Graffiti Spray (generated, `scripts/synthetic/optional_sources.py`).
* **Hard negatives and missing variety**: quiet empty rooms (hum, rumble, hiss), everyday clicks (camera
  shutters, switches, keyboards, pens, staplers) and calm neutral speech as Background Noise; horse, big cat,
  wolf and elephant calls as Animal Sound.
* Augmented copies are now made for every training recording (before, only for the generated ones).

## Adding real recordings

Put files in `incoming/<class name or slug>/` (WAV, MP3, FLAC, OGG, M4A) and optionally describe them in
`incoming/metadata.csv` (`filename, environment, device, distance_m, source, license, speaker, phrase,
source_group, notes`), then run `python -m scripts.build_dataset`. Existing Audio IDs and splits never change.
Recordings of people must be made with their consent; the help-phrase class is limited to the defined safety
phrases.
