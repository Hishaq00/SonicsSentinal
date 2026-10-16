# GTM model gtm-v8-tm-finetuned

* Network: **Teachable Machine audio network: speech-commands 18w BROWSER_FFT base (4 x Conv2D/MaxPooling2D, Dense 2000) + NewHeadDense softmax**
* Trained: 2026-09-28T14:15:46Z with `scripts/train_gtm_model.py`
* Input: 43 x 232 spectrogram of the loudest ~1 s of each 2 s segment (44.1 kHz, FFT 2048, hop 1024, 0-5 kHz), z-score normalised - computed as in static/js/gtm.js
* Training data: 49227 windows from 17046 train-split recordings (originals + augmented copies); validation 1577 and test 1581 original recordings
* Critical-class weights (folded into the output bias): `{"Glass Breaking": 2.5, "Aggression": 2.0, "Person Asking for Help": 1.5, "Explosion": 1.5}`

## Test set (recording level, exported model files)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.683 | 0.583 | 0.685 | 0.593 | 0.710 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.712 | 0.822 | 0.763 | 45 |
| Glass Breaking | 0.823 | 1.000 | 0.903 | 51 |
| Alarm or Siren | 0.923 | 0.769 | 0.839 | 156 |
| Vehicle Horn | 0.967 | 0.807 | 0.880 | 109 |
| Dog Bark | 0.839 | 0.758 | 0.797 | 62 |
| Cat | 0.769 | 0.714 | 0.741 | 14 |
| Bird | 0.757 | 0.700 | 0.727 | 40 |
| Farm Animal | 0.714 | 0.526 | 0.606 | 19 |
| Wild Animal | 0.667 | 0.909 | 0.769 | 22 |
| Insect or Frog | 0.588 | 0.385 | 0.465 | 26 |
| Gunshot | 0.835 | 0.710 | 0.768 | 100 |
| Panic Scream | 0.898 | 0.978 | 0.936 | 45 |
| Aggression | 0.581 | 0.800 | 0.673 | 45 |
| Person Asking for Help | 0.651 | 0.911 | 0.759 | 45 |
| Drone Sound | 0.763 | 1.000 | 0.865 | 45 |
| Drilling or Grinder Sound | 0.708 | 0.293 | 0.415 | 116 |
| Fireworks | 0.808 | 0.824 | 0.816 | 51 |
| Vehicle Backfire | 0.837 | 0.911 | 0.872 | 45 |
| Explosion | 0.451 | 0.822 | 0.583 | 45 |
| Door Impact | 0.830 | 0.786 | 0.807 | 56 |
| Graffiti Spray | 0.846 | 0.978 | 0.907 | 45 |
| Human Speech | 0.809 | 0.797 | 0.803 | 69 |
| Crying Baby | 0.278 | 0.833 | 0.417 | 6 |
| Human Body Sound | 0.525 | 0.583 | 0.553 | 36 |
| Laughter or Applause | 0.208 | 0.417 | 0.278 | 12 |
| Footsteps | 0.214 | 0.500 | 0.300 | 6 |
| Keyboard or Mouse | 0.444 | 0.667 | 0.533 | 18 |
| Household Appliance | 0.579 | 0.440 | 0.500 | 25 |
| Rain or Water | 0.314 | 0.344 | 0.328 | 32 |
| Wind | 0.194 | 0.667 | 0.300 | 9 |
| Thunderstorm | 0.400 | 1.000 | 0.571 | 6 |
| Fire Crackling | 0.200 | 0.667 | 0.308 | 6 |
| Church Bell | 0.176 | 1.000 | 0.300 | 6 |
| Helicopter | 0.111 | 0.167 | 0.133 | 6 |
| Train or Aircraft | 0.167 | 0.333 | 0.222 | 12 |
| Engine Running | 0.143 | 0.333 | 0.200 | 6 |
| Background Noise | 0.848 | 0.194 | 0.316 | 144 |

## Model comparison

| Model | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall (test) | Selected |
|---|---|---|---|---|---|---|
| Teachable Machine transfer (frozen speech-commands base + new softmax layer) | 0.567 | 0.482 | 0.568 | 0.488 | 0.333 |  |
| Teachable Machine network fine-tuned end to end | 0.682 | 0.606 | 0.691 | 0.597 | 0.689 |  |
| Previous export (SonicSentinel_GTM_v7) | 0.563 | 0.483 | 0.574 | 0.497 | 0.420 |  |
| Teachable Machine network fine-tuned end to end + critical-class weighting | 0.682 | 0.606 | 0.684 | 0.594 | 0.710 | yes |

## Live-microphone path (raw audio, no noise reduction or trimming)

| Split | Accuracy | Macro F1 |
|---|---|---|
| validation | 0.659 | 0.586 |
| test | 0.681 | 0.597 |

## Export check

The exported model.json / weights.bin were run with the NumPy implementation in `gtm_model/inference.py`: max |probability difference| to the trained network = 1.37e-06 on 3985 validation windows.
