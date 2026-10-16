# GTM model gtm-v6-tm-finetuned

* Network: **Teachable Machine audio network: speech-commands 18w BROWSER_FFT base (4 x Conv2D/MaxPooling2D, Dense 2000) + NewHeadDense softmax**
* Trained: 2026-09-27T12:57:52Z with `scripts/train_gtm_model.py`
* Input: 43 x 232 spectrogram of the loudest ~1 s of each 2 s segment (44.1 kHz, FFT 2048, hop 1024, 0-5 kHz), z-score normalised - computed as in static/js/gtm.js
* Training data: 25690 windows from 11012 train-split recordings (originals + augmented copies); validation 1179 and test 1181 original recordings
* Critical-class weights (folded into the output bias): `{"Glass Breaking": 1.25, "Aggression": 1.25, "Explosion": 1.1}`

## Test set (recording level, exported model files)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.801 | 0.799 | 0.864 | 0.821 | 0.760 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.603 | 0.911 | 0.726 | 45 |
| Glass Breaking | 0.882 | 1.000 | 0.938 | 45 |
| Alarm or Siren | 0.901 | 0.696 | 0.785 | 92 |
| Vehicle Horn | 0.920 | 0.920 | 0.920 | 100 |
| Animal Sound | 0.835 | 0.865 | 0.850 | 111 |
| Gunshot | 0.916 | 0.760 | 0.831 | 100 |
| Panic Scream | 0.860 | 0.956 | 0.905 | 45 |
| Aggression | 0.778 | 0.933 | 0.848 | 45 |
| Person Asking for Help | 0.741 | 0.956 | 0.835 | 45 |
| Drone Sound | 0.900 | 1.000 | 0.947 | 45 |
| Drilling or Grinder Sound | 0.774 | 0.625 | 0.691 | 104 |
| Fireworks | 0.796 | 0.956 | 0.869 | 45 |
| Vehicle Backfire | 0.746 | 0.978 | 0.846 | 45 |
| Explosion | 0.536 | 0.822 | 0.649 | 45 |
| Door Impact | 0.741 | 0.860 | 0.796 | 50 |
| Graffiti Spray | 0.880 | 0.978 | 0.926 | 45 |
| Background Noise | 0.776 | 0.477 | 0.591 | 174 |

## Model comparison

| Model | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall (test) | Selected |
|---|---|---|---|---|---|---|
| Teachable Machine transfer (frozen speech-commands base + new softmax layer) | 0.688 | 0.713 | 0.696 | 0.717 | 0.511 |  |
| Teachable Machine network fine-tuned end to end | 0.817 | 0.841 | 0.804 | 0.823 | 0.790 |  |
| Previous export (SonicSentinel_GTM_v5) | 0.686 | 0.718 | 0.694 | 0.717 | 0.578 |  |
| Teachable Machine network fine-tuned end to end + critical-class weighting | 0.817 | 0.840 | 0.803 | 0.823 | 0.770 | yes |

## Live-microphone path (raw audio, no noise reduction or trimming)

| Split | Accuracy | Macro F1 |
|---|---|---|
| validation | 0.815 | 0.838 |
| test | 0.798 | 0.816 |

## Export check

The exported model.json / weights.bin were run with the NumPy implementation in `gtm_model/inference.py`: max |probability difference| to the trained network = 1.04e-06 on 2693 validation windows.
