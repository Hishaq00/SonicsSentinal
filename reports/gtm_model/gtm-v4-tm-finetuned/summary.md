# GTM model gtm-v4-tm-finetuned

* Network: **Teachable Machine audio network: speech-commands 18w BROWSER_FFT base (4 x Conv2D/MaxPooling2D, Dense 2000) + NewHeadDense softmax**
* Trained: 2026-09-26T11:04:32Z with `scripts/train_gtm_model.py`
* Input: 43 x 232 spectrogram of the loudest ~1 s of each 2 s segment (44.1 kHz, FFT 2048, hop 1024, 0-5 kHz), z-score normalised - computed as in static/js/gtm.js
* Training data: 12290 windows from 5541 train-split recordings (originals + augmented copies); validation 739 and test 741 original recordings
* Critical-class weights (folded into the output bias): `{"Glass Breaking": 1.5, "Aggression": 1.1}`

## Test set (recording level, exported model files)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.865 | 0.869 | 0.883 | 0.871 | 0.911 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.900 | 0.800 | 0.847 | 90 |
| Glass Breaking | 0.957 | 1.000 | 0.978 | 45 |
| Alarm or Siren | 0.969 | 0.674 | 0.795 | 92 |
| Vehicle Horn | 0.905 | 0.950 | 0.927 | 100 |
| Animal Sound | 0.830 | 0.876 | 0.852 | 89 |
| Gunshot | 0.970 | 0.970 | 0.970 | 100 |
| Panic Scream | 0.915 | 0.956 | 0.935 | 45 |
| Aggression | 0.788 | 0.911 | 0.845 | 45 |
| Person Asking for Help | 0.786 | 0.978 | 0.871 | 45 |
| Background Noise | 0.667 | 0.711 | 0.688 | 90 |

## Model comparison

| Model | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall (test) | Selected |
|---|---|---|---|---|---|---|
| Teachable Machine transfer (frozen speech-commands base + new softmax layer) | 0.752 | 0.764 | 0.779 | 0.772 | 0.711 |  |
| Teachable Machine network fine-tuned end to end | 0.880 | 0.885 | 0.865 | 0.872 | 0.911 |  |
| Previous export (SonicSentinel_AcousticX_GTM_v3) | 0.345 | 0.347 | 0.379 | 0.385 | 0.300 |  |
| Teachable Machine network fine-tuned end to end + critical-class weighting | 0.881 | 0.886 | 0.865 | 0.871 | 0.911 | yes |

## Live-microphone path (raw audio, no noise reduction or trimming)

| Split | Accuracy | Macro F1 |
|---|---|---|
| validation | 0.877 | 0.882 |
| test | 0.876 | 0.884 |

## Export check

The exported model.json / weights.bin were run with the NumPy implementation in `gtm_model/inference.py`: max |probability difference| to the trained network = 7.45e-07 on 1633 validation windows.
