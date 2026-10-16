# GTM model gtm-v9-tm-finetuned

* Network: **Teachable Machine audio network: speech-commands 18w BROWSER_FFT base (4 x Conv2D/MaxPooling2D, Dense 2000) + NewHeadDense softmax**
* Trained: 2026-09-28T17:35:28Z with `scripts/train_gtm_model.py`
* Input: 43 x 232 spectrogram of the loudest ~1 s of each 2 s segment (44.1 kHz, FFT 2048, hop 1024, 0-5 kHz), z-score normalised - computed as in static/js/gtm.js
* Training data: 33252 windows from 13443 train-split recordings (originals + augmented copies); validation 1810 and test 1803 original recordings
* Critical-class weights (folded into the output bias): `{}`

## Test set (recording level, exported model files)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.679 | 0.657 | 0.824 | 0.714 | 0.622 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.644 | 0.844 | 0.731 | 45 |
| Glass Breaking | 0.847 | 0.980 | 0.909 | 51 |
| Alarm or Siren | 0.820 | 0.846 | 0.833 | 156 |
| Vehicle Horn | 0.778 | 0.835 | 0.805 | 109 |
| Animal Sound | 0.796 | 0.699 | 0.745 | 246 |
| Gunshot | 0.669 | 0.830 | 0.741 | 100 |
| Panic Scream | 0.688 | 0.978 | 0.807 | 45 |
| Aggression | 0.452 | 0.844 | 0.589 | 45 |
| Person Asking for Help | 0.368 | 0.867 | 0.517 | 45 |
| Drone Sound | 0.714 | 1.000 | 0.833 | 45 |
| Drilling or Grinder Sound | 0.542 | 0.621 | 0.579 | 124 |
| Fireworks | 0.667 | 0.863 | 0.752 | 51 |
| Vehicle Backfire | 0.606 | 0.956 | 0.741 | 45 |
| Explosion | 0.337 | 0.622 | 0.438 | 45 |
| Door Impact | 0.588 | 0.839 | 0.691 | 56 |
| Graffiti Spray | 0.818 | 1.000 | 0.900 | 45 |
| Background Noise | 0.826 | 0.380 | 0.521 | 550 |

## Model comparison

| Model | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall (test) | Selected |
|---|---|---|---|---|---|---|
| Teachable Machine transfer (frozen speech-commands base + new softmax layer) | 0.604 | 0.626 | 0.599 | 0.629 | 0.510 |  |
| Teachable Machine network fine-tuned end to end | 0.694 | 0.725 | 0.680 | 0.715 | 0.622 | yes |

## Live-microphone path (raw audio, no noise reduction or trimming)

| Split | Accuracy | Macro F1 |
|---|---|---|
| validation | 0.699 | 0.720 |
| test | 0.691 | 0.725 |

## Export check

The exported model.json / weights.bin were run with the NumPy implementation in `gtm_model/inference.py`: max |probability difference| to the trained network = 1.61e-06 on 4651 validation windows.
