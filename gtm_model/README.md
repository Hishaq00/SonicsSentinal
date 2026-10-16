# GTM model (Google Teachable Machine audio model)

## What is here

| Path | Contents |
|---|---|
| `model/model.json`, `model/weights.bin`, `model/metadata.json` | the exported TensorFlow.js model the app uses (Teachable Machine audio format, 17 labels in the same order as `config/classes.json`) |
| `base_model/` | Google's speech-commands 18w `BROWSER_FFT` base model (downloaded by the training script) |
| `pipeline.py` | turns an audio file / live window into exactly the input the browser computes (`static/js/gtm.js`) |
| `inference.py` | NumPy runner for exported TF.js models (used to check the export and by the comparison report) |
| `export_gtm_samples.py` | exports ~1 s training samples per class from the **training split only** |

## How the app uses it (placement / URL configuration)

The model runs **in the browser** (TF.js). Admin → Models → *Teachable Machine* chooses the source:

* **Local files** (default): the app serves `gtm_model/model/`. To replace the model, copy the three exported
  files (`model.json`, `weights.bin`, `metadata.json`) into `gtm_model/model/`.
* **Hosted link**: paste the Teachable Machine share link
  (`https://teachablemachine.withgoogle.com/models/<id>/`). The choice is saved in the settings and the audit
  trail.

The labels must be exactly the class names of the Python model (`config/classes.json`); the app rejects a model
whose labels differ
(`tests/test_gtm.py::test_gtm_label_mismatch_is_rejected`).

## How the current model was trained

```bash
pip install -r requirements-gtm.txt   # PyTorch, training only
python -m scripts.train_gtm_model     # writes gtm_model/model/ and reports/gtm_model/<version>/
```

The script builds the Teachable Machine audio network (speech-commands base + `NewHeadDense` output layer),
trains it on the same recordings, splits, pre-processing and 2-second segments as the Python model, selects
on validation, evaluates on the test split and exports in Teachable Machine format. Results:
`reports/gtm_model/gtm-v4-tm-finetuned/summary.md` (86.5 % test accuracy).

## Creating the Teachable Machine project (SRS evidence)

The SRS asks for a GTM Audio Project with screenshots of all classes. To create it from the same training data:

1. `python -m gtm_model.export_gtm_samples` → `gtm_model/training_samples/<Class>/…wav` (one or two ~1 s windows
   per training recording, 44.1 kHz) and `_concat/<Class>.wav` (all samples of a class back to back).
2. Open <https://teachablemachine.withgoogle.com/train/audio>, create the 17 classes with exactly these names:
   Machinery Fault, Glass Breaking, Alarm or Siren, Vehicle Horn, Animal Sound, Gunshot, Panic Scream,
   Aggression, Person Asking for Help, Drone Sound, Drilling or Grinder Sound, Fireworks, Vehicle Backfire,
   Explosion, Door Impact, Graffiti Spray, Background Noise. Teachable Machine's first, built-in class is already
   called "Background Noise" — use it for our Background Noise class (`_concat/Background Noise.wav`).
3. For each class, play `_concat/<Class>.wav` into the microphone recorder (or use a virtual audio cable) and
   extract the samples.
4. Train, then take screenshots of every class, the training settings and a test prediction; save them in
   `screenshots/gtm/`. Export the model ("Upload my model" gives the share link; "Download" gives the files).
5. Put the link in the README and in Admin → Models. Keep `gtm_model/model/` (the fine-tuned model) as the
   app's model unless the website-trained model is better on the validation split — the website trains only the
   output layer, which reached 77.9 % in our comparison.
