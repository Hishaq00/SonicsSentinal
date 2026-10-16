# GTM model gtm-v7-tm-transfer

* Network: **Teachable Machine audio network: speech-commands 18w BROWSER_FFT base (4 x Conv2D/MaxPooling2D, Dense 2000) + NewHeadDense softmax**
* Trained: 2026-09-28T13:22:23Z with `scripts/train_gtm_model.py`
* Input: 43 x 232 spectrogram of the loudest ~1 s of each 2 s segment (44.1 kHz, FFT 2048, hop 1024, 0-5 kHz), z-score normalised - computed as in static/js/gtm.js
* Training data: 49227 windows from 17046 train-split recordings (originals + augmented copies); validation 1577 and test 1581 original recordings
* Critical-class weights (folded into the output bias): `{"Glass Breaking": 1.25, "Gunshot": 1.1, "Panic Scream": 1.1, "Explosion": 3.0}`

## Test set (recording level, exported model files)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.574 | 0.488 | 0.582 | 0.497 | 0.420 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.724 | 0.467 | 0.568 | 45 |
| Glass Breaking | 0.894 | 0.824 | 0.857 | 51 |
| Alarm or Siren | 0.787 | 0.686 | 0.733 | 156 |
| Vehicle Horn | 0.838 | 0.615 | 0.709 | 109 |
| Dog Bark | 0.868 | 0.532 | 0.660 | 62 |
| Cat | 0.471 | 0.571 | 0.516 | 14 |
| Bird | 0.451 | 0.575 | 0.505 | 40 |
| Farm Animal | 0.268 | 0.579 | 0.367 | 19 |
| Wild Animal | 0.692 | 0.818 | 0.750 | 22 |
| Insect or Frog | 0.480 | 0.462 | 0.471 | 26 |
| Gunshot | 0.677 | 0.420 | 0.519 | 100 |
| Panic Scream | 0.833 | 0.889 | 0.860 | 45 |
| Aggression | 0.614 | 0.600 | 0.607 | 45 |
| Person Asking for Help | 0.619 | 0.578 | 0.598 | 45 |
| Drone Sound | 0.623 | 0.956 | 0.754 | 45 |
| Drilling or Grinder Sound | 0.614 | 0.371 | 0.462 | 116 |
| Fireworks | 0.590 | 0.706 | 0.643 | 51 |
| Vehicle Backfire | 0.542 | 0.867 | 0.667 | 45 |
| Explosion | 0.386 | 0.600 | 0.470 | 45 |
| Door Impact | 0.683 | 0.768 | 0.723 | 56 |
| Graffiti Spray | 0.721 | 0.978 | 0.830 | 45 |
| Human Speech | 0.690 | 0.710 | 0.700 | 69 |
| Crying Baby | 0.333 | 0.833 | 0.476 | 6 |
| Human Body Sound | 0.405 | 0.417 | 0.411 | 36 |
| Laughter or Applause | 0.189 | 0.583 | 0.286 | 12 |
| Footsteps | 0.136 | 0.500 | 0.214 | 6 |
| Keyboard or Mouse | 0.270 | 0.556 | 0.364 | 18 |
| Household Appliance | 0.308 | 0.160 | 0.211 | 25 |
| Rain or Water | 0.360 | 0.281 | 0.316 | 32 |
| Wind | 0.286 | 0.667 | 0.400 | 9 |
| Thunderstorm | 0.462 | 1.000 | 0.632 | 6 |
| Fire Crackling | 0.148 | 0.667 | 0.242 | 6 |
| Church Bell | 0.160 | 0.667 | 0.258 | 6 |
| Helicopter | 0.083 | 0.167 | 0.111 | 6 |
| Train or Aircraft | 0.100 | 0.083 | 0.091 | 12 |
| Engine Running | 0.050 | 0.167 | 0.077 | 6 |
| Background Noise | 0.705 | 0.215 | 0.330 | 144 |

## Model comparison

| Model | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall (test) | Selected |
|---|---|---|---|---|---|---|
| Teachable Machine transfer (frozen speech-commands base + new softmax layer) | 0.567 | 0.482 | 0.568 | 0.488 | 0.333 |  |
| Teachable Machine transfer (frozen speech-commands base + new softmax layer) + critical-class weighting | 0.562 | 0.480 | 0.564 | 0.488 | 0.440 | yes |

## Live-microphone path (raw audio, no noise reduction or trimming)

| Split | Accuracy | Macro F1 |
|---|---|---|
| validation | 0.550 | 0.472 |
| test | 0.576 | 0.497 |

## Export check

The exported model.json / weights.bin were run with the NumPy implementation in `gtm_model/inference.py`: max |probability difference| to the trained network = 1.97e-06 on 3985 validation windows.
