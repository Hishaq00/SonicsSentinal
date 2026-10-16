# GTM model gtm-v5-tm-transfer

* Network: **Teachable Machine audio network: speech-commands 18w BROWSER_FFT base (4 x Conv2D/MaxPooling2D, Dense 2000) + NewHeadDense softmax**
* Trained: 2026-09-27T12:55:28Z with `scripts/train_gtm_model.py`
* Input: 43 x 232 spectrogram of the loudest ~1 s of each 2 s segment (44.1 kHz, FFT 2048, hop 1024, 0-5 kHz), z-score normalised - computed as in static/js/gtm.js
* Training data: 25690 windows from 11012 train-split recordings (originals + augmented copies); validation 1179 and test 1181 original recordings
* Critical-class weights (folded into the output bias): `{"Glass Breaking": 1.1, "Gunshot": 1.1, "Panic Scream": 1.25, "Person Asking for Help": 1.1, "Explosion": 2.5}`

## Test set (recording level, exported model files)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.693 | 0.714 | 0.743 | 0.717 | 0.578 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.429 | 0.800 | 0.558 | 45 |
| Glass Breaking | 1.000 | 0.978 | 0.989 | 45 |
| Alarm or Siren | 0.753 | 0.598 | 0.667 | 92 |
| Vehicle Horn | 0.804 | 0.780 | 0.792 | 100 |
| Animal Sound | 0.748 | 0.775 | 0.761 | 111 |
| Gunshot | 0.708 | 0.630 | 0.667 | 100 |
| Panic Scream | 0.889 | 0.889 | 0.889 | 45 |
| Aggression | 0.620 | 0.689 | 0.653 | 45 |
| Person Asking for Help | 0.475 | 0.622 | 0.538 | 45 |
| Drone Sound | 0.894 | 0.933 | 0.913 | 45 |
| Drilling or Grinder Sound | 0.670 | 0.587 | 0.626 | 104 |
| Fireworks | 0.764 | 0.933 | 0.840 | 45 |
| Vehicle Backfire | 0.629 | 0.867 | 0.729 | 45 |
| Explosion | 0.317 | 0.578 | 0.409 | 45 |
| Door Impact | 0.762 | 0.640 | 0.696 | 50 |
| Graffiti Spray | 0.952 | 0.889 | 0.920 | 45 |
| Background Noise | 0.731 | 0.437 | 0.547 | 174 |

## Model comparison

| Model | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall (test) | Selected |
|---|---|---|---|---|---|---|
| Teachable Machine transfer (frozen speech-commands base + new softmax layer) | 0.688 | 0.713 | 0.696 | 0.717 | 0.511 |  |
| Teachable Machine transfer (frozen speech-commands base + new softmax layer) + critical-class weighting | 0.677 | 0.708 | 0.691 | 0.716 | 0.578 | yes |

## Live-microphone path (raw audio, no noise reduction or trimming)

| Split | Accuracy | Macro F1 |
|---|---|---|
| validation | 0.662 | 0.697 |
| test | 0.675 | 0.698 |

## Export check

The exported model.json / weights.bin were run with the NumPy implementation in `gtm_model/inference.py`: max |probability difference| to the trained network = 2.15e-06 on 2693 validation windows.
