# Python model py-v9-hgb-weighted

* Algorithm: **Histogram Gradient Boosting + critical-class weighting**
* Trained: 2026-09-28T12:19:01Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"base_model": "hist_gradient_boosting", "base_params": {"l2_regularization": 0.0244, "learning_rate": 0.0837, "max_leaf_nodes": 63, "min_samples_leaf": 40}, "class_weights": {"Glass Breaking": 2.5, "Gunshot": 3.0, "Panic Scream": 3.0, "Aggression": 2.0, "Person Asking for Help": 3.0, "Explosion": 1.1}, "tuned_on": "validation split", "validation_effect": {"before": {"critical_recall_min": 0.4666666666666667, "f1_macro": 0.662493208865759}, "after": {"critical_recall_min": 0.6666666666666666, "f1_macro": 0.6753359564015567}}}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.753 | 0.655 | 0.710 | 0.673 | 0.667 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.872 | 0.911 | 0.891 | 45 |
| Glass Breaking | 0.941 | 0.941 | 0.941 | 51 |
| Alarm or Siren | 0.897 | 0.897 | 0.897 | 156 |
| Vehicle Horn | 0.938 | 0.826 | 0.878 | 109 |
| Dog Bark | 0.727 | 0.774 | 0.750 | 62 |
| Cat | 0.571 | 0.571 | 0.571 | 14 |
| Bird | 0.500 | 0.675 | 0.574 | 40 |
| Farm Animal | 0.722 | 0.684 | 0.703 | 19 |
| Wild Animal | 0.800 | 0.909 | 0.851 | 22 |
| Insect or Frog | 0.519 | 0.538 | 0.528 | 26 |
| Gunshot | 0.856 | 0.770 | 0.811 | 100 |
| Panic Scream | 0.851 | 0.889 | 0.870 | 45 |
| Aggression | 0.850 | 0.756 | 0.800 | 45 |
| Person Asking for Help | 0.667 | 0.667 | 0.667 | 45 |
| Drone Sound | 0.978 | 1.000 | 0.989 | 45 |
| Drilling or Grinder Sound | 0.756 | 0.586 | 0.660 | 116 |
| Fireworks | 0.875 | 0.824 | 0.848 | 51 |
| Vehicle Backfire | 0.919 | 0.756 | 0.829 | 45 |
| Explosion | 0.685 | 0.822 | 0.747 | 45 |
| Door Impact | 0.695 | 0.732 | 0.713 | 56 |
| Graffiti Spray | 0.978 | 0.978 | 0.978 | 45 |
| Human Speech | 0.819 | 0.855 | 0.837 | 69 |
| Crying Baby | 0.462 | 1.000 | 0.632 | 6 |
| Human Body Sound | 0.535 | 0.639 | 0.582 | 36 |
| Laughter or Applause | 0.500 | 0.500 | 0.500 | 12 |
| Footsteps | 0.083 | 0.167 | 0.111 | 6 |
| Keyboard or Mouse | 0.625 | 0.556 | 0.588 | 18 |
| Household Appliance | 0.545 | 0.720 | 0.621 | 25 |
| Rain or Water | 0.452 | 0.594 | 0.514 | 32 |
| Wind | 0.583 | 0.778 | 0.667 | 9 |
| Thunderstorm | 0.750 | 1.000 | 0.857 | 6 |
| Fire Crackling | 0.500 | 0.833 | 0.625 | 6 |
| Church Bell | 0.333 | 0.500 | 0.400 | 6 |
| Helicopter | 0.100 | 0.167 | 0.125 | 6 |
| Train or Aircraft | 0.312 | 0.417 | 0.357 | 12 |
| Engine Running | 0.300 | 0.500 | 0.375 | 6 |
| Background Noise | 0.722 | 0.542 | 0.619 | 144 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Histogram Gradient Boosting | 0.662 | 0.662 | 0.746 | 0.667 | 0.511 |  |
| Histogram Gradient Boosting + critical-class weighting | 0.672 | 0.672 | 0.753 | 0.673 | 0.667 | yes |

## Live-microphone rendering (same recordings, as the live monitor delivers them)

| Split | Accuracy | Macro F1 | Min critical recall |
|---|---|---|---|
| validation | 0.734 | 0.655 | 0.689 |
| test | 0.754 | 0.677 | 0.600 |

## Confidence calibration

Temperature 0.9374 fitted on the validation segments. Expected calibration error on the test segments 0.029 -> 0.0263, mean confidence 0.6826 -> 0.7065 against accuracy 0.693. The predicted class does not change.
